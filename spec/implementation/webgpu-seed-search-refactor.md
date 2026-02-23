# WebGPU Seed Search Simplification Plan

## 1. 背景と課題認識
- `src/lib/webgpu/seed-search/runner.ts` がデバイス初期化・ディスパッチ計画・バッファ管理・進捗管理を一括抱え、状態が複雑化。
- `search-worker-webgpu.ts` 側でも独自の状態機構を持ち、`runner` と二重管理になっている。
- `message-encoder.ts` など SearchConditions に対する薄いラッパーが乱立し、入力正規化のロジックが分散。
- GPU ディスパッチ単位が不透明で、`MatchOutputBuffer` や `maxComputeWorkgroupsPerDimension` など WebGPU 固有制約に沿った計画が見えにくい。
- pause/resume/stop/abort の組み合わせが `runner` と Worker で重複実装され、バグ温床になっている。

> **更新 (2025-11-20)**: `runner.ts` / `batch-planner.ts` / `buffer-pool.ts` / `message-encoder.ts` は削除済み。以下の内容は完了済みリファクタの設計記録として維持する。

## 2. 目標
1. 入力 (`SearchConditions`, `targetSeeds`) → 出力 (`InitialSeedResult[]`) を最短経路で実行できる処理フローを定義。
2. 責務を「UI/メインスレッド」「Worker」「SeedSearchEngine(WebGPU)」に分割し、状態管理を単層化。
3. GPU デバイス制約 (workgroup/dispatch 上限・バッファ容量) を前提に、セグメント/ディスパッチ計画を determinisitic に生成。
4. 中断・一時停止・再開・停止の UX を `parallel-search-worker.ts` と同様の API に合わせる。

## 3. 目標外 (Non-goals)
- `.wgsl` (SHA1 kernel) のアルゴリズム変更。
- 既存 wasm ベース parallel worker の挙動変更。
- UI/検索条件入力フォームの刷新。

## 4. 新処理フロー概要
```
UI thread
  └─ prepareSearchJob(SearchConditions, targetSeeds)
          ↓ SeedSearchJob { segments[], targetSeedsU32, summary }
Worker (webgpu)
  └─ SeedSearchController
        loop segments:
          - honor pause/stop
          - engine.executeSegment(segment)
          - forward RESULT / PROGRESS
SeedSearchEngine (WebGPU)
  ├─ initOnce(): pipeline/buffers/workgroup
  ├─ ensureTargetBuffer(targetSeeds)
  ├─ writeConfig(segment.configWords)
  ├─ dispatchWorkgroups(messageCount)
  └─ readMatchBuffer() → InitialSeedResult[]
```

### 4.1 Component Responsibilities
| Component | 主責務 |
| --- | --- |
| `prepareSearchJob` | SearchConditions 正規化 / `resolveTimePlan` / GPU 制約に基づく segment 分割 / configWords 生成 |
| Worker (`search-worker-webgpu.ts`) | ジョブ受理、`SeedSearchController` の状態管理、UI へのメッセージ転送 |
| `SeedSearchController` | Segment イテレーション、pause/resume/stop、進捗計算、`SeedSearchEngine` 呼び出し |
| `SeedSearchEngine` | WebGPU パイプライン初期化と再利用、単一 segment の dispatch/結果読み出し |

### 4.2 Segment と Dispatch
- **Segment** = { `timeWindow`, `timer0Range`, `vcountRange`, `keyMask`, `messageCount`, `configWords` }。
- `prepareSearchJob` が `resolveTimePlan` の allowed 秒を走査し、GPU 上限 (`maxWorkgroupsPerDimension`, `matchBufferCapacity`) を満たす `messageCount` で分割。
- segment 内部では `.wgsl` が global index から timer0/vcount/秒を計算するため、追加の per-message データは不要。

### 4.3 進捗計算
- `parallel-search-worker.ts` と同じ `countAllowedSecondsInInterval` を利用。
- `SeedSearchController` が segment 開始/終了で `processedSeconds` を更新し、0.5s 間隔で `PROGRESS` を送信。

### 4.4 Worker state machine
```
IDLE → RUNNING → {PAUSED ↔ RUNNING} → STOPPING → IDLE
```
- `pause()` で `isPaused = true`、segment 間 or dispatch 待機中に `await resume`。
- `stop()` / `AbortSignal` で `shouldStop = true`、ループを抜け `STOPPED` を通知。

## 5. 実装タスク計画
### Phase 0: 事前整備
1. `src/lib/webgpu/seed-search/message-encoder.ts` 等の不要なラッパーを削除。
2. `resolveTimePlan` / `iterateAllowedSubChunks` を webgpu モジュールからも再利用できるように export 調整。

### Phase 1: 入力正規化の一本化
1. `prepareSearchJob.ts` を新規追加。
2. `SearchConditions` → `SeedSearchJob` (segments[], targetSeedsUint32, summary) の生成実装。
3. GPU 制約の取得 (`adapter.limits`, マッチバッファ容量) と `maxMessagesPerDispatch` 計算ロジックを追加。
4. 単体テスト (条件の端数や 0 秒範囲を確認)。

### Phase 2: Worker/Controller 再構築
1. `search-worker-webgpu.ts` を `SeedSearchController` ベースに書き換え。
   - `workerState` を簡素化 (`isRunning`, `isPaused`, `shouldStop`).
   - `timerState` を `parallel-search-worker.ts` から再利用。
2. `SeedSearchController` 実装 (`run(job)`, `pause()`, `resume()`, `stop()`).
3. 既存 UI からのメッセージ型は互換に保つ (START/PAUSE/RESUME/STOP)。

### Phase 3: SeedSearchEngine (WebGPU) 実装
1. `seed-search-engine.ts` を追加。
   - `init()`, `executeSegment(segment, targetSeeds)`, `dispose()`。
   - パイプライン/バッファ初期化、`targetBuffer` サイズ管理、`matchBuffer` map/unmap。
2. `runner.ts` と `batch-planner.ts` から必要なロジックを移行し、不要部分を削除。
3. GPU 制約を `SeedSearchEngine` 内でも検証し、`prepareSearchJob` の想定と齟齬があればエラー返却。

### Phase 4: 結果ストリーミングと完了処理
1. `executeSegment` を async generator で実装し、Worker が逐次 `RESULT` を送信。
2. 最終進捗と統計 (`matchesFound`, `elapsedTime`) をまとめて `COMPLETE` 通知。
3. 異常停止/Abort 時のリソース解放と `STOPPED` メッセージ整備。

### Phase 5: 移行と後処理
1. 旧 `runner.ts` / `batch-planner.ts` / `buffer-pool.ts` を削除。
2. 影響範囲 (`search-worker-webgpu`, UI, テスト) をアップデート。
3. `spec/implementation` のドキュメントに実装結果を追記し、QA/ベンチ手順を整理。

## 6. リスクと緩和策
| リスク | 緩和策 |
| --- | --- |
| GPU 限界値が環境により変動 | `prepareSearchJob` で adapter.limits を読取り、fallback 既定値を設定。 |
| 進捗報告の遅延 | segment 処理中も `setInterval` 相当の tick で報告。 |
| UI 側の API 変更 | Worker メッセージ構造は現行と互換に保つ。 |

## 7. 成功指標
- 主要 API が `prepareSearchJob` + Worker 呼び出しの 2 ステップのみで済む。
- `search-worker-webgpu.ts` のコード行数が 40% 以上削減。
- `src/lib/webgpu/seed-search/` 配下の総コード行数が旧実装比で明確に減少し、責務集約が可視化される。

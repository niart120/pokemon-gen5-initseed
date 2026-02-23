# WebGPU 初期Seed検索 実装設計

## 背景と目的
- WebGPU を活用した初期Seed検索を本番アプリに組み込み、WASM より高いスループットを確保する。
- 既存ロジックとの循環依存や速度低下を避けるため、WebGPU パスは独立モジュールとして実装する。
- GPU 側でメッセージ生成から SHA-1 計算までを完結させ、CPU 側の大規模バッファ生成を排除する。

## 非機能要件
- GPU が利用不可の場合は自動で既存パスへフォールバック。
- バッチサイズはデバイスリミットとホストメモリ（96MiB 既定）を超えない。
- Double buffering が常に有効化されるバッチ計画を採用。
- 既存 Worker メッセージプロトコル（READY/PROGRESS/RESULT 等）を維持。

## モジュール構成
```
src/lib/webgpu/seed-search/
   ├─ prepare-search-job.ts     // SearchConditions → SeedSearchJob 変換
   ├─ seed-search-controller.ts // 進捗管理・CPU検証・コールバック集約
   ├─ seed-search-engine.ts     // WebGPU Compute 実行とバッファ管理
   ├─ device-context.ts         // Adapter/Device 取得とリミット情報
   ├─ (shared) search/time/time-plan.ts // 日時レンジの正規化・オフセット取得
   ├─ constants.ts / types.ts   // 共有定数と型定義
   ├─ kernel/
   │    ├─ sha1-generate.wgsl       // メッセージ生成 + SHA-1 計算シェーダ
   │    └─ seed-search-kernel.ts    // Kernelモジュール生成と BindGroupLayout 定義
```
- コントローラは `run` / `pause` / `resume` / `stop` を公開し、`SeedSearchJob` を逐次消化する。
- `SeedSearchEngine` が GPU 実行とリソース確保を担当し、コントローラ経由で Web Worker コールバックへ結果を返す。

## ワーカー連携
- `src/workers/search-worker-webgpu.ts` は `prepareSearchJob` で `SeedSearchJob` を構築し、`SeedSearchController` に委譲する。
- `SearchWorkerManager` は設定ストアからモードを判定し、WebGPU モード時のみ専用 worker を初期化する設計を維持。
- 既存 worker と同一メッセージ型 (`WorkerRequest`/`WorkerResponse`) を再利用し、進捗・結果・停止イベントを統一的に通知する。
- エラー発生時はメッセージ経由で通知し、マネージャ側で CPU/WASM パスにフォールバックする。

## チャンク化・バッチ処理
1. `prepareSearchJob(conditions, targetSeeds)` が Timer0×VCount×日時レンジを `SeedSearchJobSegment` 配列へ分割。
   - セグメントは keyCode / timer0 範囲 / vcount 範囲 / 時間オフセットを保持し、GPU へ渡す `configWords` を生成する。
   - `SeedSearchJobSummary` と `SeedSearchJobLimits` により、UI とエンジン双方が同じメタ情報を参照可能。
2. Dispatch 上限は `prepareSearchJob` 内で WebGPU リミット (`maxComputeWorkgroupsPerDimension`, `maxStorageBufferBindingSize`) とターゲット容量から決定。
   - 値は `SeedSearchJobLimits` に格納され、エンジン初期化および進捗推定に利用される。
3. コントローラはセグメントを順次実行し、各ディスパッチ後に `WorkerProgressMessage` を発行する。

## フォールバック戦略
- Worker 初期化エラー、デバイス非対応、実行時 `GPUValidationError` が発生した場合、`ERROR` メッセージ送信後に自動で CPU/WASM パスへ切り替える。
- ユーザー設定で WebGPU 無効化した場合は、worker を生成せず従来パスで開始。

## プロファイリングと進捗報告
- `profiling.ts` で `totalMs`、`dispatchMs`、`readbackMs`、バッチ数などを集計し、`PROGRESS` メッセージ経由で UI に共有。
- 進捗の粒度はチャンク単位 + 各 dispatch 後の累計メッセージ数。
- 必要に応じて詳細ログをフラグで切り替えられるようにする。

## テスト計画
- **ユニット**: `prepare-search-job.test.ts` による時間計画・キー入力フィルタの検証、`gpu-message-mapping.test.ts` で GPU/CPU メッセージ整合性を担保。
- **統合 (Vitest Browser)**: WebGPU worker 経由での検索実行ベンチを追加し、CPU/WASM 結果と比較。
- **E2E**: `public/test-integration.html` で UI からの検索・中断・再開とフォールバック経路を検証。
- **非対応環境**: WebGPU 未対応ブラウザでフォールバックが選択されることを確認。

## 実装ステップ
1. 共通設定・ストアに WebGPU モード判定とフラグを追加。
2. `SearchWorkerManager` をリファクタし、GPU worker 起動経路を実装。
3. WebGPU ランナーおよび周辺ユーティリティを段階的に実装。
4. Worker からのフォールバック処理を整備。
5. テスト・ドキュメントを更新し、手動検証シナリオを整備。

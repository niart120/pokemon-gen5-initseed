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
  ├─ runner.ts                 // WebGPU Seed 検索ランナー公開API
  ├─ device-context.ts         // Adapter/Device 取得とリミット情報
  ├─ batch-planner.ts          // バッチ上限・チャンク分割ロジック
  ├─ message-encoder.ts        // GPU-friendly チャンク記述子生成
  ├─ buffers/
  │    ├─ buffer-pool.ts       // 二重バッファ資源管理
  ├─ pipelines/
  │    ├─ sha1-generate.wgsl   // メッセージ生成 + SHA-1 計算シェーダ
  │    ├─ pipeline-factory.ts  // ComputePipeline 構築
  ├─ profiling.ts              // 計測構造体・統計整形
```
- ランナーは `init`, `supports`, `runSearch`, `dispose` の4メソッドを公開する。
- `runSearch` は `AbortSignal` を受け取り、中断指示や STOP メッセージに対応。

## ワーカー連携
- `src/workers/search-worker-webgpu.ts` を新設し、上記ランナーをラップする。
- `SearchWorkerManager` は設定ストアからモードを判定し、WebGPU モード時は専用 worker を初期化。
- 既存 worker と同一メッセージ型 (`WorkerRequest`/`WorkerResponse`) を再利用する。
- エラー発生時はメッセージ経由で通知し、マネージャ側で CPU/WASM パスにフォールバック。

## チャンク化・バッチ処理
1. `prepareWorkload(conditions)` で Timer0×VCount×秒の検索範囲をチャンク列へ分割。
   - チャンクは「連続 Timer0 範囲」「連続 VCount 範囲」「連続秒範囲」で構成。
   - 各チャンクは GPU シェーダへ渡す `GeneratedConfig` と `baseOffset`・`messageCount` を保持。
2. `BatchPlanner` がデバイスリミット (`maxStorageBufferBindingSize`, workgroup 上限) とホストメモリを参照し、
   - 1ディスパッチ当たりの最大メッセージ数を算出。
   - Double buffering が成立するよう、チャンク内 dispatch 件数 >= 2 を保証。
3. 実行時は `computeGenerated` 相当のフローを踏襲し、各チャンクをストリーミング dispatch。

## フォールバック戦略
- Worker 初期化エラー、デバイス非対応、実行時 `GPUValidationError` が発生した場合、`ERROR` メッセージ送信後に自動で CPU/WASM パスへ切り替える。
- ユーザー設定で WebGPU 無効化した場合は、worker を生成せず従来パスで開始。

## プロファイリングと進捗報告
- `profiling.ts` で `totalMs`、`dispatchMs`、`readbackMs`、バッチ数などを集計し、`PROGRESS` メッセージ経由で UI に共有。
- 進捗の粒度はチャンク単位 + 各 dispatch 後の累計メッセージ数。
- 必要に応じて詳細ログをフラグで切り替えられるようにする。

## テスト計画
- **ユニット**: BatchPlanner と message encoder の境界値テスト、フォールバック動作テスト。
- **統合 (Vitest Browser)**: WebGPU worker 経由での検索実行ベンチを追加し、CPU/WASM 結果と比較。
- **E2E**: `public/test-integration.html` に WebGPU ルートを追加し、UI からの検索・中断・再開を検証。
- **非対応環境**: WebGPU 未対応ブラウザでのフォールバック確認。

## 実装ステップ
1. 共通設定・ストアに WebGPU モード判定とフラグを追加。
2. `SearchWorkerManager` をリファクタし、GPU worker 起動経路を実装。
3. WebGPU ランナーおよび周辺ユーティリティを段階的に実装。
4. Worker からのフォールバック処理を整備。
5. テスト・ドキュメントを更新し、手動検証シナリオを整備。

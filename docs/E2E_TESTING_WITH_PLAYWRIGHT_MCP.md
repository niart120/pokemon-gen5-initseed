# E2Eテスト実行手順 - Playwright-MCP版

## 概要

このドキュメントでは、Playwright-MCPを使用してポケモンBW/BW2初期Seed探索アプリケーションのE2Eテストを実行する手順を説明します。

従来のJavaScriptテストページでの手動テストをPlaywright-MCPで自動化し、実際のユーザーワークフローを検証します。

## 前提条件

- Playwright-MCPが有効であること
- 開発サーバーが起動していること (`npm run dev`)
- WebAssemblyモジュールが正常にビルドされていること

## テスト構成

### 基本テスト環境
- **URL**: `http://localhost:5173/`
- **ROM設定**: Black (B) / Japan (JPN) / Nintendo DS
- **WebAssembly**: 32並列Worker環境
- **検証対象**: 実際のプロダクションアプリケーション

## テストケース一覧

### Test Case 1: 基本アプリケーション動作確認

**目的**: アプリケーションの基本機能とWebAssembly統合の確認

**手順**:
1. アプリケーション起動確認
2. WebAssembly初期化確認
3. UI要素の表示確認
4. 基本設定の確認

**Playwright-MCP実行例**:
```javascript
// ページナビゲーション
await mcp_playwright_browser_navigate({ url: "http://localhost:5173/" });

// ページスナップショット取得
await mcp_playwright_browser_snapshot();

// WebAssembly初期化ログ確認
await mcp_playwright_browser_wait_for({ time: 2 });
const messages = await mcp_playwright_browser_console_messages();
// "WebAssembly acceleration enabled!" の確認
```

**期待結果**:
- ページが正常に読み込まれる
- WebAssemblyモジュールが初期化される
- 全UI要素が表示される
- コンソールエラーがない

### Test Case 2: 並列探索の実行確認

**目的**: 32並列Workerでの実際の探索処理の動作確認

**テスト条件**:
- ROM: Black (B), Region: Japan (JPN), Hardware: Nintendo DS
- Target Seeds: `0x400899a7`, `0xbc3a30e8`, `0x958e4e88`
- Worker数: 32
- 日付範囲: 2000/01/01 12:00:00 ～ 2099/12/31 12:01:59

**Playwright-MCP実行例**:
```javascript
// Start Searchボタンクリック
await mcp_playwright_browser_click({
  element: "Start Search button",
  ref: "e543"
});

// 進捗監視（3秒間）
await mcp_playwright_browser_wait_for({ time: 3 });

// 現在の状況確認
await mcp_playwright_browser_snapshot();

// Stopボタンクリック
await mcp_playwright_browser_click({
  element: "Stop button", 
  ref: "e257"
});
```

**期待結果**:
- 32個のWorkerが正常に初期化される
- WebAssemblyモジュールが各Workerで読み込まれる
- 進捗率が増加する（例: 20.5% → 42.9%）
- 処理速度が表示される（例: 138,586,910/秒）
- 停止操作で正常に終了する

### Test Case 3: 特定Seed検証テスト

**目的**: 既知の結果を持つSeedに対する検索精度確認

**テストデータ**:
```javascript
const VERIFICATION_SEEDS = [
  {
    seed: 0x14B11BA6,
    expected: { datetime: '2066/06/27 03:02:48', timer0: 0xC79 }
  },
  {
    seed: 0x8A30480D, 
    expected: { datetime: '2063/11/23 11:39:47', timer0: 0xC79 }
  },
  {
    seed: 0x9E02B0AE,
    expected: { datetime: '2073/08/30 03:55:06', timer0: 0xC7A }
  },
  {
    seed: 0xADFA2178,
    expected: { datetime: '2072/06/21 13:22:13', timer0: 0xC7A }
  }
];
```

**Playwright-MCP実行例**:
```javascript
// Target Seedsフィールドに検証用Seedを入力
await mcp_playwright_browser_type({
  element: "Target Seeds input field",
  ref: "e179",
  text: "0x14B11BA6\n0x8A30480D\n0x9E02B0AE\n0xADFA2178"
});

// 日付範囲を該当期間に設定
await mcp_playwright_browser_type({
  element: "Start Year input",
  ref: "e100", 
  text: "2060"
});

await mcp_playwright_browser_type({
  element: "End Year input",
  ref: "e122",
  text: "2080"
});

// 探索実行
await mcp_playwright_browser_click({
  element: "Start Search button",
  ref: "e543"
});

// 結果が見つかるまで待機（またはタイムアウト）
await mcp_playwright_browser_wait_for({ time: 30 });

// 結果確認
await mcp_playwright_browser_snapshot();
```

**期待結果**:
- 各Seedに対して期待される日時・Timer0値が検出される
- 検索結果が正確に表示される
- エラーが発生しない

### Test Case 4: 重複Seed検証テスト

**目的**: 同一Seedに対する複数解の検出確認

**テストデータ**:
```javascript
const DUPLICATE_SEED_TEST = {
  seed: 0xFC4AA3AC,
  expectedResults: [
    { datetime: '2025/10/18 02:48:49', timer0: 0xC7A },
    { datetime: '2041/05/25 17:17:59', timer0: 0xC7A }
  ]
};
```

**Playwright-MCP実行例**:
```javascript
// Target SeedにFC4AA3ACを設定
await mcp_playwright_browser_type({
  element: "Target Seeds input field",
  ref: "e179",
  text: "0xFC4AA3AC"
});

// 探索実行と結果確認
await mcp_playwright_browser_click({
  element: "Start Search button", 
  ref: "e543"
});

// 複数解が見つかるまで待機
await mcp_playwright_browser_wait_for({ time: 60 });

// 結果の複数性確認
await mcp_playwright_browser_snapshot();
```

**期待結果**:
- 同一Seedに対して複数の日時・Timer0値の組み合わせが検出される
- 重複検出機能が正常に動作する

### Test Case 5: UI操作・機能テスト

**目的**: ユーザーインターフェースの各機能の動作確認

**テスト項目**:
- ROM設定変更
- 日付・時刻設定
- Worker数調整
- 結果フィルタリング
- エクスポート機能

**Playwright-MCP実行例**:
```javascript
// ROM設定変更
await mcp_playwright_browser_click({
  element: "ROM Version dropdown",
  ref: "e41"
});

await mcp_playwright_browser_click({
  element: "White (W) option",
  ref: "option-w"
});

// Worker数調整
await mcp_playwright_browser_click({
  element: "Worker count slider",
  ref: "e201"
});

// 設定変更の反映確認
await mcp_playwright_browser_snapshot();
```

**期待結果**:
- 各UI要素が正常に動作する
- 設定変更が即座に反映される
- アクセシビリティが保たれている

## 統合テスト実行手順

### 1. 環境準備
```bash
# 開発サーバー起動
npm run dev

# WebAssemblyビルド確認
npm run build:wasm
```

### 2. Playwright-MCPでの基本動作確認
```javascript
// ブラウザでアプリケーション起動
await mcp_playwright_browser_navigate({ 
  url: "http://localhost:5173/" 
});

// 初期状態確認
await mcp_playwright_browser_snapshot();
```

### 3. WebAssembly統合確認
```javascript
// コンソールメッセージでWebAssembly初期化確認
const messages = await mcp_playwright_browser_console_messages();
// "WebAssembly acceleration enabled!" を確認
```

### 4. 実際の探索実行
```javascript
// 探索開始
await mcp_playwright_browser_click({
  element: "Start Search button",
  ref: "e543"
});

// 進捗監視
await mcp_playwright_browser_wait_for({ time: 5 });

// 停止・状態確認
await mcp_playwright_browser_click({
  element: "Stop button",
  ref: "e257"
});
```

### 5. 結果検証
```javascript
// 最終状態確認
await mcp_playwright_browser_snapshot();

// エラーがないことの確認
const finalMessages = await mcp_playwright_browser_console_messages();
```

## パフォーマンス指標

### 正常動作の基準
- **Worker初期化**: 32個全て3秒以内
- **WebAssembly読み込み**: 各Worker 1秒以内
- **探索速度**: 100,000,000/秒以上
- **メモリ使用量**: 安定（リークなし）
- **UI応答性**: 操作に対して即座に反応

### 異常検出基準
- コンソールエラーの発生
- Worker初期化失敗
- WebAssembly読み込み失敗
- 探索速度の著しい低下
- UI操作の無応答

## トラブルシューティング

### よくある問題と対処法

**WebAssembly読み込み失敗**:
```bash
npm run build:wasm
npm run copy:wasm
```

**Worker初期化失敗**:
- ブラウザコンソールでエラー確認
- WebAssemblyファイルの存在確認
- CORS設定確認

**探索速度低下**:
- システムリソース確認
- 他のプロセス確認
- Worker数調整

## 実行ログ例

### 正常実行時のログ
```
✅ WebAssembly module loaded successfully
✅ All workers initialized and started  
🚀 Starting parallel search with 32 workers
✅ Worker 0: WebAssembly initialized
...
✅ Worker 31: WebAssembly initialized
📊 Progress: 42.9% (2,706,048,000 / 6,311,347,440)
⚡ Rate: 138,586,910/s
🛑 Search stopped by user
```

### 異常時のログ例
```
❌ Failed to load WebAssembly module
❌ Worker initialization timeout
⚠️ Performance degradation detected
```

## まとめ

このPlaywright-MCP版E2Eテストにより、従来の手動テストを自動化し、継続的なアプリケーション品質確保が可能になります。

特に、実際のユーザーワークフローをWebAssembly統合環境で検証することで、本格的なE2Eテストが実現されています。

# EggGenerationPanel 実装ガイド

## 1. 実装準備

### 1.1 必要な知識
- TypeScript strict mode
- React function components
- Zustand 状態管理
- Web Workers
- WebAssembly (wasm-bindgen)
- Vitest テストフレームワーク

### 1.2 開発環境セットアップ
```bash
# 依存関係インストール
npm install

# WASM ビルド (wasm-pack が必要)
npm run build:wasm

# 開発サーバー起動
npm run dev

# テスト実行
npm run test
```

## 2. Phase 1: 型定義とWorker基盤

### 2.1 型定義ファイル作成

#### ステップ 1: `src/types/egg.ts` 作成
```bash
# ファイル作成
touch src/types/egg.ts
```

仕様書の「2. データ型定義」セクションに従って実装:
- 基本型 (ParentRole, EverstonePlan, IvSet, etc.)
- パラメータ型 (EggGenerationParams, EggGenerationParamsHex)
- Worker通信型 (EggWorkerRequest, EggWorkerResponse)
- バリデーション関数 (validateEggParams)
- 型ガード関数 (isEggWorkerResponse)

**重要事項:**
- すべての enum 値は WASM の Rust 実装と一致させる
- BigInt と Number の変換に注意
- Unknown IV (32) の取り扱い

#### ステップ 2: 型テスト作成 `src/types/egg.test.ts`
```typescript
import { describe, it, expect } from 'vitest';
import {
  validateEggParams,
  hexParamsToEggParams,
  type EggGenerationParamsHex,
} from './egg';

describe('egg types', () => {
  it('hexParamsToEggParams converts correctly', () => {
    const hex: EggGenerationParamsHex = {
      baseSeedHex: 'FFFFFFFFFFFFFFFF',
      userOffsetHex: '0',
      count: 100,
      // ... 他のフィールド
    };
    const params = hexParamsToEggParams(hex);
    expect(params.baseSeed).toBe(BigInt('0xFFFFFFFFFFFFFFFF'));
    expect(params.userOffset).toBe(BigInt(0));
  });

  it('validateEggParams detects invalid count', () => {
    const params = {
      // ... valid fields
      count: 100001, // over limit
    };
    const errors = validateEggParams(params as any);
    expect(errors.length).toBeGreaterThan(0);
    expect(errors[0]).toContain('count');
  });
});
```

### 2.2 Worker実装

#### ステップ 1: `src/workers/egg-worker.ts` 作成

仕様書の「3. Worker実装」セクションに従って実装:

**重要なポイント:**
1. WASM初期化の適切な処理
2. EggSeedEnumerator の正しい使用方法
3. メモリ管理 (`.free()` 呼び出し)
4. エラーハンドリング
5. stop リクエストの処理

**WASM オブジェクト構築例:**
```typescript
// ParentsIVs
const parentsIVs = new wasm.ParentsIVs();
parentsIVs.male = params.parents.male;
parentsIVs.female = params.parents.female;

// GenerationConditions
const conditions = new wasm.GenerationConditions();
conditions.has_nidoran_flag = params.conditions.hasNidoranFlag;
// ... 他のフィールド設定

// 使用後は必ず解放
try {
  // ... enumerator 使用
} finally {
  enumerator.free();
  conditions.free();
  parentsIVs.free();
}
```

#### ステップ 2: Worker統合テスト `src/test/egg/egg-worker.test.ts`
```typescript
import { describe, it, expect, beforeAll } from 'vitest';
import { initWasmForTesting } from '@/test/wasm-loader';

describe('egg-worker integration', () => {
  beforeAll(async () => {
    await initWasmForTesting();
  });

  it('should enumerate eggs with basic params', async () => {
    // Worker を作成してテスト
    const worker = new Worker(
      new URL('@/workers/egg-worker.ts', import.meta.url),
      { type: 'module' }
    );

    const results: any[] = [];
    let completed = false;

    worker.onmessage = (ev) => {
      const msg = ev.data;
      if (msg.type === 'RESULTS') {
        results.push(...msg.payload.results);
      } else if (msg.type === 'COMPLETE') {
        completed = true;
      }
    };

    worker.postMessage({
      type: 'START_GENERATION',
      params: {
        baseSeed: BigInt('0x1234567890ABCDEF'),
        userOffset: BigInt(0),
        count: 10,
        // ... 他のパラメータ
      },
    });

    // 完了まで待機
    await new Promise((resolve) => {
      const check = setInterval(() => {
        if (completed) {
          clearInterval(check);
          resolve(null);
        }
      }, 100);
    });

    expect(results.length).toBeGreaterThan(0);
    worker.terminate();
  });
});
```

### 2.3 WorkerManager実装

#### ステップ 1: `src/lib/egg/egg-worker-manager.ts` 作成

仕様書の「4. WorkerManager実装」セクションに従って実装:

**重要なポイント:**
1. Worker のライフサイクル管理
2. コールバックの適切な配信
3. エラーハンドリング
4. 状態管理 (idle/running/stopping)

**使用パターン:**
```typescript
const manager = new EggWorkerManager();

manager
  .onResults((payload) => {
    console.log('Got results:', payload.results);
  })
  .onComplete((completion) => {
    console.log('Completed:', completion);
  })
  .onError((message, category, fatal) => {
    console.error('Error:', message);
  });

await manager.start(params);
```

#### ステップ 2: WorkerManager単体テスト `src/lib/egg/egg-worker-manager.test.ts`
```typescript
import { describe, it, expect, vi } from 'vitest';
import { EggWorkerManager } from './egg-worker-manager';

describe('EggWorkerManager', () => {
  it('should start and handle results', async () => {
    const mockWorker = {
      postMessage: vi.fn(),
      onmessage: null,
      onerror: null,
      terminate: vi.fn(),
    };

    const createWorker = () => mockWorker as any;
    const manager = new EggWorkerManager(createWorker);

    const resultsCb = vi.fn();
    const completeCb = vi.fn();

    manager.onResults(resultsCb).onComplete(completeCb);

    const params = {
      // ... valid params
    };

    await manager.start(params);

    expect(mockWorker.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'START_GENERATION' })
    );
  });
});
```

## 3. Phase 2: 状態管理

### 3.1 Zustandストア作成

#### ステップ 1: `src/store/egg-store.ts` 作成

仕様書の「5.2 状態管理 (Zustand)」セクションに従って実装:

**重要なポイント:**
1. ドラフトパラメータと確定パラメータの分離
2. バリデーション統合
3. WorkerManager との連携
4. 結果の蓄積

**ストア使用例:**
```typescript
const {
  draftParams,
  updateDraftParams,
  validateDraft,
  startGeneration,
  stopGeneration,
  results,
  status,
} = useEggStore();

// パラメータ更新
updateDraftParams({ count: 500 });

// バリデーション
validateDraft();

// 開始
await startGeneration();
```

#### ステップ 2: ストアテスト `src/store/egg-store.test.ts`
```typescript
import { describe, it, expect } from 'vitest';
import { useEggStore } from './egg-store';

describe('egg-store', () => {
  it('should update draft params', () => {
    const { updateDraftParams, draftParams } = useEggStore.getState();
    updateDraftParams({ count: 200 });
    expect(useEggStore.getState().draftParams.count).toBe(200);
  });

  it('should validate params', () => {
    const { updateDraftParams, validateDraft, validationErrors } = useEggStore.getState();
    updateDraftParams({ count: -1 }); // invalid
    validateDraft();
    expect(useEggStore.getState().validationErrors.length).toBeGreaterThan(0);
  });
});
```

## 4. Phase 3: UIコンポーネント

### 4.1 パネルレイアウト

#### `src/components/egg/EggGenerationPanel.tsx`
仕様書の「5.3 パネル実装サンプル」に従って実装:
- レスポンシブレイアウト (isStack判定)
- 2カラムレイアウト (デスクトップ)
- 縦積みレイアウト (モバイル)

### 4.2 パラメータカード

#### `src/components/egg/EggParamsCard.tsx`
```typescript
export const EggParamsCard: React.FC = () => {
  const { draftParams, updateDraftParams } = useEggStore();
  const locale = useLocale();

  return (
    <PanelCard title="タマゴ生成パラメータ">
      {/* 基本設定セクション */}
      <div>
        <Label>初期Seed</Label>
        <Input
          value={draftParams.baseSeedHex}
          onChange={(e) => updateDraftParams({ baseSeedHex: e.target.value })}
        />
      </div>

      {/* 親個体情報セクション */}
      <div>
        <Label>♂親 IV</Label>
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <Input
            key={i}
            type="number"
            min={0}
            max={32}
            value={draftParams.parents.male[i]}
            onChange={(e) => {
              const newMale = [...draftParams.parents.male];
              newMale[i] = Number(e.target.value);
              updateDraftParams({
                parents: { ...draftParams.parents, male: newMale as IvSet },
              });
            }}
          />
        ))}
      </div>

      {/* 他のセクション... */}
    </PanelCard>
  );
};
```

**UI設計のポイント:**
1. 各入力フィールドは適切なバリデーション
2. 16進数入力は即座に正規化
3. IV入力は 0-32 の範囲制限
4. ツールチップで説明を提供

### 4.3 フィルターカード

#### `src/components/egg/EggFilterCard.tsx`
```typescript
export const EggFilterCard: React.FC = () => {
  const { draftParams, updateDraftParams } = useEggStore();

  const handleIvRangeChange = (statIndex: number, minMax: 'min' | 'max', value: number) => {
    const newRanges = [...(draftParams.filter?.ivRanges || [])];
    newRanges[statIndex] = {
      ...newRanges[statIndex],
      [minMax]: value,
    };
    updateDraftParams({
      filter: {
        ...(draftParams.filter || {}),
        ivRanges: newRanges as any,
      },
    });
  };

  return (
    <PanelCard title="フィルター設定">
      {/* IV範囲スライダー */}
      {['HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe'].map((stat, i) => (
        <div key={i}>
          <Label>{stat}</Label>
          <div className="flex gap-2">
            <input
              type="range"
              min={0}
              max={32}
              value={draftParams.filter?.ivRanges[i]?.min ?? 0}
              onChange={(e) => handleIvRangeChange(i, 'min', Number(e.target.value))}
            />
            <input
              type="range"
              min={0}
              max={32}
              value={draftParams.filter?.ivRanges[i]?.max ?? 32}
              onChange={(e) => handleIvRangeChange(i, 'max', Number(e.target.value))}
            />
          </div>
        </div>
      ))}

      {/* 性格フィルター */}
      <Select
        value={draftParams.filter?.nature?.toString()}
        onValueChange={(v) => updateDraftParams({
          filter: {
            ...(draftParams.filter || {}),
            nature: v ? Number(v) : undefined,
          },
        })}
      >
        <SelectTrigger>
          <SelectValue placeholder="性格を選択" />
        </SelectTrigger>
        <SelectContent>
          {NATURES.map((name, i) => (
            <SelectItem key={i} value={i.toString()}>{name}</SelectItem>
          ))}
        </SelectContent>
      </Select>

      {/* 他のフィルター... */}
    </PanelCard>
  );
};
```

### 4.4 実行制御カード

#### `src/components/egg/EggRunCard.tsx`
```typescript
export const EggRunCard: React.FC = () => {
  const {
    status,
    validationErrors,
    validateDraft,
    startGeneration,
    stopGeneration,
    results,
    lastCompletion,
  } = useEggStore();

  const handleStart = async () => {
    validateDraft();
    if (validationErrors.length === 0) {
      await startGeneration();
    }
  };

  const isRunning = status === 'running';
  const canStart = status === 'idle' || status === 'completed' || status === 'error';

  return (
    <PanelCard title="実行制御">
      <div className="flex gap-2">
        <Button
          onClick={handleStart}
          disabled={!canStart}
        >
          <Play size={16} />
          開始
        </Button>
        <Button
          onClick={stopGeneration}
          disabled={!isRunning}
        >
          <Square size={16} />
          停止
        </Button>
      </div>

      {/* ステータス表示 */}
      <div>
        <Label>ステータス</Label>
        <p>{status}</p>
      </div>

      {/* 進捗表示 */}
      {lastCompletion && (
        <div>
          <p>処理済み: {lastCompletion.processedCount}</p>
          <p>フィルター適用後: {lastCompletion.filteredCount}</p>
          <p>実行時間: {lastCompletion.elapsedMs.toFixed(0)}ms</p>
        </div>
      )}

      {/* バリデーションエラー表示 */}
      {validationErrors.length > 0 && (
        <div className="text-red-500">
          {validationErrors.map((err, i) => (
            <p key={i}>{err}</p>
          ))}
        </div>
      )}
    </PanelCard>
  );
};
```

### 4.5 結果表示カード

#### `src/components/egg/EggResultsCard.tsx`
```typescript
export const EggResultsCard: React.FC = () => {
  const { results } = useEggStore();

  return (
    <PanelCard title="生成結果">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th>Advance</th>
              <th>HP</th>
              <th>Atk</th>
              <th>Def</th>
              <th>SpA</th>
              <th>SpD</th>
              <th>Spe</th>
              <th>性格</th>
              <th>性別</th>
              <th>特性</th>
              <th>色違い</th>
              <th>めざパ</th>
              <th>PID</th>
            </tr>
          </thead>
          <tbody>
            {results.map((row, i) => (
              <tr key={i}>
                <td>{row.advance}</td>
                {row.egg.ivs.map((iv, j) => (
                  <td key={j}>{iv === 32 ? '?' : iv}</td>
                ))}
                <td>{NATURES[row.egg.nature]}</td>
                <td>{row.egg.gender}</td>
                <td>{row.egg.ability}</td>
                <td>{SHINY_TYPES[row.egg.shiny]}</td>
                <td>
                  {row.egg.hiddenPower.type === 'known'
                    ? `${HP_TYPES[row.egg.hiddenPower.hpType]} ${row.egg.hiddenPower.power}`
                    : '?'}
                </td>
                <td>{row.egg.pid.toString(16).toUpperCase()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </PanelCard>
  );
};
```

## 5. Phase 4: 統合とテスト

### 5.1 WASM統合テスト

#### `src/test/egg/egg-wasm-integration.test.ts`
```typescript
import { describe, it, expect, beforeAll } from 'vitest';
import { initWasmForTesting } from '@/test/wasm-loader';
import { getWasm } from '@/lib/core/wasm-interface';

describe('egg WASM integration', () => {
  beforeAll(async () => {
    await initWasmForTesting();
  });

  it('should create EggSeedEnumerator and enumerate eggs', () => {
    const wasm = getWasm();
    const { EggSeedEnumerator, ParentsIVs, GenerationConditions, TrainerIds, GenderRatio } = wasm;

    const parents = new ParentsIVs();
    parents.male = [31, 31, 31, 31, 31, 31];
    parents.female = [31, 31, 31, 31, 31, 31];

    const conditions = new GenerationConditions();
    conditions.has_nidoran_flag = false;
    conditions.everstone = wasm.EverstonePlan.None();
    conditions.uses_ditto = false;
    conditions.allow_hidden_ability = false;
    conditions.female_parent_has_hidden = false;
    conditions.reroll_count = 0;
    conditions.trainer_ids = new TrainerIds(1234, 5678);
    conditions.gender_ratio = new GenderRatio(127, false);

    const enumerator = new EggSeedEnumerator(
      BigInt('0x1234567890ABCDEF'),
      BigInt(0),
      10,
      conditions,
      parents,
      null, // no filter
      false, // no NPC
      1 // BwContinue
    );

    const results = [];
    while (true) {
      const data = enumerator.next_egg();
      if (!data) break;
      results.push(data);
    }

    expect(results.length).toBeGreaterThan(0);
    expect(results[0].advance).toBe(0);

    enumerator.free();
    conditions.free();
    parents.free();
  });

  it('should apply filter correctly', () => {
    const wasm = getWasm();
    // ... フィルター適用テスト
  });
});
```

### 5.2 E2Eテスト (Playwright)

#### `src/test/e2e/egg-panel.spec.ts`
```typescript
import { test, expect } from '@playwright/test';

test('egg panel - basic workflow', async ({ page }) => {
  await page.goto('http://localhost:5173');

  // EggGenerationPanel タブに移動 (実装に応じて)
  await page.click('text=タマゴ生成');

  // パラメータ入力
  await page.fill('[data-testid="egg-base-seed"]', '1234567890ABCDEF');
  await page.fill('[data-testid="egg-count"]', '100');

  // 親IV入力
  for (let i = 0; i < 6; i++) {
    await page.fill(`[data-testid="egg-male-iv-${i}"]`, '31');
    await page.fill(`[data-testid="egg-female-iv-${i}"]`, '31');
  }

  // 開始ボタンクリック
  await page.click('[data-testid="egg-start-button"]');

  // 結果が表示されるまで待機
  await page.waitForSelector('[data-testid="egg-results-table"]', { timeout: 10000 });

  // 結果行が存在することを確認
  const rows = await page.locator('[data-testid="egg-result-row"]').count();
  expect(rows).toBeGreaterThan(0);
});
```

## 6. デバッグとトラブルシューティング

### 6.1 よくある問題

#### WASM初期化エラー
```typescript
// エラー: "EggSeedEnumerator not exposed in WASM"
// 解決: wasm-pkg/src/lib.rs で EggSeedEnumerator がエクスポートされているか確認
```

#### メモリリーク
```typescript
// 問題: WASMオブジェクトが解放されない
// 解決: try-finally ブロックで必ず .free() 呼び出し
try {
  const enumerator = new wasm.EggSeedEnumerator(...);
  // ... use enumerator
} finally {
  enumerator.free();
}
```

#### Worker通信エラー
```typescript
// 問題: BigInt がシリアライズされない
// 解決: BigInt を Number または String に変換
const advance = Number(bigintValue); // or bigintValue.toString()
```

### 6.2 デバッグツール

#### Console ログ
```typescript
// Worker内
console.log('[EggWorker] Starting enumeration', params);

// Manager内
console.log('[EggWorkerManager] Received results', payload.results.length);
```

#### Chrome DevTools
- Sources タブで Worker コードにブレークポイント設定
- Network タブで WASM ファイル読み込み確認
- Memory タブでメモリリーク検出

## 7. パフォーマンス最適化

### 7.1 バッチ処理
```typescript
// 結果を一定個数ごとにまとめて送信
const BATCH_SIZE = 100;
const batch: EnumeratedEggData[] = [];

while (true) {
  const data = enumerator.next_egg();
  if (!data) break;
  
  batch.push(parseEnumeratedEggData(data));
  
  if (batch.length >= BATCH_SIZE) {
    postResults(batch);
    batch.length = 0; // clear
  }
}

if (batch.length > 0) {
  postResults(batch);
}
```

### 7.2 メモリ管理
```typescript
// 大量結果はストリーミング処理
// 全件保持せず、表示分のみ保持
const MAX_DISPLAY_RESULTS = 10000;

manager.onResults((payload) => {
  set((state) => {
    const newResults = [...state.results, ...payload.results];
    // 上限を超えたら古い結果を削除
    if (newResults.length > MAX_DISPLAY_RESULTS) {
      return {
        results: newResults.slice(-MAX_DISPLAY_RESULTS),
      };
    }
    return { results: newResults };
  });
});
```

## 8. 国際化対応

### 8.1 ラベル定義

#### `src/lib/i18n/strings/egg-params.ts`
```typescript
import type { LocaleString } from './types';

export const eggParamsBaseSeedLabel: LocaleString = {
  ja: '初期Seed',
  en: 'Initial Seed',
};

export const eggParamsCountLabel: LocaleString = {
  ja: '列挙上限',
  en: 'Max Count',
};

// ... 他のラベル
```

### 8.2 使用例
```typescript
import { useLocale } from '@/lib/i18n/locale-context';
import { eggParamsBaseSeedLabel } from '@/lib/i18n/strings/egg-params';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';

const label = resolveLocaleValue(eggParamsBaseSeedLabel, locale);
```

## 9. チェックリスト

### 実装完了チェックリスト
- [ ] 型定義ファイル作成 (`src/types/egg.ts`)
- [ ] Worker実装 (`src/workers/egg-worker.ts`)
- [ ] WorkerManager実装 (`src/lib/egg/egg-worker-manager.ts`)
- [ ] Zustandストア作成 (`src/store/egg-store.ts`)
- [ ] UIコンポーネント作成
  - [ ] EggGenerationPanel
  - [ ] EggParamsCard
  - [ ] EggFilterCard
  - [ ] EggRunCard
  - [ ] EggResultsCard
- [ ] 単体テスト作成
  - [ ] 型テスト
  - [ ] WorkerManagerテスト
  - [ ] ストアテスト
- [ ] 統合テスト作成
  - [ ] Worker統合テスト
  - [ ] WASM統合テスト
- [ ] E2Eテスト作成
- [ ] 国際化対応
- [ ] ドキュメント更新

### コード品質チェックリスト
- [ ] ESLint エラーなし
- [ ] Prettier フォーマット適用
- [ ] TypeScript strict mode 準拠
- [ ] すべてのテストがパス
- [ ] メモリリークなし
- [ ] WASM オブジェクトの適切な解放
- [ ] エラーハンドリング実装
- [ ] ローディング状態の適切な表示

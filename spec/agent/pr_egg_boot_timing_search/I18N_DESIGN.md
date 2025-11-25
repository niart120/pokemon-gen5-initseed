# i18n 国際化対応設計

## 1. 概要

Search(Egg) パネルの多言語対応（日本語・英語）設計を定義する。既存の `lib/i18n/` パターンを踏襲し、新規コンポーネント向けの国際化文字列を追加する。

## 2. 既存パターン分析

### 2.1 i18n アーキテクチャ

```
src/lib/i18n/
├── locale-context.tsx    # React Context Provider
├── locales.ts            # ロケール定数エクスポート
└── strings/              # 各機能別の文字列定義
    ├── types.ts          # 型定義
    ├── common.ts         # 共通文字列
    ├── egg-run.ts        # タマゴ生成制御
    ├── egg-params.ts     # タマゴパラメータ
    ├── search-control.ts # 検索制御
    └── ...
```

### 2.2 型定義パターン

```typescript
// src/lib/i18n/strings/types.ts

import type { SupportedLocale } from '@/types/i18n';

/**
 * 単純な言語別テキスト
 */
export type LocaleText = Record<SupportedLocale, string>;

/**
 * 言語別マップ（オブジェクト値）
 */
export type LocaleMap<T> = Record<SupportedLocale, T>;

/**
 * ロケール値解決
 */
export function resolveLocaleValue<T>(map: LocaleMap<T>, locale: SupportedLocale): T {
  return map[locale];
}
```

### 2.3 使用パターン

```tsx
// コンポーネント内での使用
import { useLocale } from '@/lib/i18n/locale-context';
import { eggRunPanelTitle, eggRunButtonLabels } from '@/lib/i18n/strings/egg-run';

export const EggRunCard: React.FC = () => {
  const locale = useLocale();
  
  return (
    <PanelCard title={eggRunPanelTitle[locale]}>
      <Button>{eggRunButtonLabels.start[locale]}</Button>
    </PanelCard>
  );
};
```

## 3. Search(Egg) パネル向け文字列定義

### 3.1 ファイル構成

```
src/lib/i18n/strings/
└── egg-boot-timing-search.ts    # 新規作成
```

### 3.2 文字列定義

```typescript
// src/lib/i18n/strings/egg-boot-timing-search.ts

import type { SupportedLocale } from '@/types/i18n';
import type { LocaleText, LocaleMap } from './types';
import type { EggBootTimingSearchStatus } from '@/store/egg-boot-timing-search-store';
import type { EggBootTimingCompletion } from '@/types/egg-boot-timing-search';

// ===========================
// パネルタイトル
// ===========================

export const eggBootTimingSearchPanelTitle: LocaleText = {
  ja: 'Search (Egg)',
  en: 'Search (Egg)',
};

export const eggBootTimingSearchPanelDescription: LocaleText = {
  ja: '起動時間条件から孵化個体を検索',
  en: 'Search for hatched Pokémon by boot timing conditions',
};

// ===========================
// パラメータ入力ラベル
// ===========================

export const eggBootTimingParamsLabels = {
  // 日時関連
  startDatetime: {
    ja: '開始日時',
    en: 'Start Date/Time',
  } satisfies LocaleText,
  rangeSeconds: {
    ja: '検索範囲（秒）',
    en: 'Search Range (seconds)',
  } satisfies LocaleText,
  
  // Timer0/VCount
  timer0Range: {
    ja: 'Timer0 範囲',
    en: 'Timer0 Range',
  } satisfies LocaleText,
  vcountRange: {
    ja: 'VCount 範囲',
    en: 'VCount Range',
  } satisfies LocaleText,
  
  // キー入力
  keyInput: {
    ja: 'キー入力',
    en: 'Key Input',
  } satisfies LocaleText,
  
  // 時刻フィルター
  timeRangeFilter: {
    ja: '時刻範囲フィルター',
    en: 'Time Range Filter',
  } satisfies LocaleText,
  hourRange: {
    ja: '時',
    en: 'Hour',
  } satisfies LocaleText,
  minuteRange: {
    ja: '分',
    en: 'Minute',
  } satisfies LocaleText,
  secondRange: {
    ja: '秒',
    en: 'Second',
  } satisfies LocaleText,
  
  // 消費範囲
  userOffset: {
    ja: '開始 Advance',
    en: 'Start Advance',
  } satisfies LocaleText,
  advanceCount: {
    ja: '検索件数',
    en: 'Advance Count',
  } satisfies LocaleText,
  
  // 結果上限
  maxResults: {
    ja: '結果上限',
    en: 'Max Results',
  } satisfies LocaleText,
};

// ===========================
// 実行制御ラベル
// ===========================

export const eggBootTimingRunPanelTitle: LocaleText = {
  ja: '検索制御',
  en: 'Search Control',
};

export const eggBootTimingRunButtonLabels = {
  start: {
    ja: '検索開始',
    en: 'Start Search',
  } satisfies LocaleText,
  starting: {
    ja: '開始中...',
    en: 'Starting...',
  } satisfies LocaleText,
  stop: {
    ja: '停止',
    en: 'Stop',
  } satisfies LocaleText,
  stopping: {
    ja: '停止中...',
    en: 'Stopping...',
  } satisfies LocaleText,
  pause: {
    ja: '一時停止',
    en: 'Pause',
  } satisfies LocaleText,
  resume: {
    ja: '再開',
    en: 'Resume',
  } satisfies LocaleText,
};

export const eggBootTimingRunStatusLabels: LocaleMap<Record<EggBootTimingSearchStatus, string>> = {
  ja: {
    idle: 'アイドル',
    starting: '開始中',
    running: '検索中',
    paused: '一時停止',
    stopping: '停止中',
    completed: '完了',
    error: 'エラー',
  },
  en: {
    idle: 'Idle',
    starting: 'Starting',
    running: 'Searching',
    paused: 'Paused',
    stopping: 'Stopping',
    completed: 'Completed',
    error: 'Error',
  },
};

// ===========================
// 進捗表示
// ===========================

export const eggBootTimingProgressLabels = {
  elapsedTime: {
    ja: '経過時間',
    en: 'Elapsed Time',
  } satisfies LocaleText,
  remainingTime: {
    ja: '残り時間',
    en: 'Remaining Time',
  } satisfies LocaleText,
  processed: {
    ja: '処理済み',
    en: 'Processed',
  } satisfies LocaleText,
  found: {
    ja: '発見数',
    en: 'Found',
  } satisfies LocaleText,
  activeWorkers: {
    ja: 'アクティブ Worker',
    en: 'Active Workers',
  } satisfies LocaleText,
  progress: {
    ja: '進捗',
    en: 'Progress',
  } satisfies LocaleText,
};

// ===========================
// 結果表示
// ===========================

export const eggBootTimingResultsPanelTitle: LocaleText = {
  ja: '検索結果',
  en: 'Search Results',
};

export const eggBootTimingResultsColumnLabels = {
  bootDatetime: {
    ja: '起動日時',
    en: 'Boot Date/Time',
  } satisfies LocaleText,
  timer0: {
    ja: 'Timer0',
    en: 'Timer0',
  } satisfies LocaleText,
  vcount: {
    ja: 'VCount',
    en: 'VCount',
  } satisfies LocaleText,
  keyInput: {
    ja: 'キー入力',
    en: 'Key Input',
  } satisfies LocaleText,
  lcgSeed: {
    ja: 'LCG Seed',
    en: 'LCG Seed',
  } satisfies LocaleText,
  advance: {
    ja: 'Advance',
    en: 'Advance',
  } satisfies LocaleText,
  ivs: {
    ja: '個体値',
    en: 'IVs',
  } satisfies LocaleText,
  nature: {
    ja: '性格',
    en: 'Nature',
  } satisfies LocaleText,
  ability: {
    ja: '特性',
    en: 'Ability',
  } satisfies LocaleText,
  gender: {
    ja: '性別',
    en: 'Gender',
  } satisfies LocaleText,
  shiny: {
    ja: '色違い',
    en: 'Shiny',
  } satisfies LocaleText,
  hiddenPower: {
    ja: 'めざパ',
    en: 'Hidden Power',
  } satisfies LocaleText,
  stable: {
    ja: '安定',
    en: 'Stable',
  } satisfies LocaleText,
};

export const eggBootTimingResultsEmptyMessage: LocaleText = {
  ja: '検索結果がありません',
  en: 'No results found',
};

export const eggBootTimingResultsCountLabel: LocaleText = {
  ja: '件',
  en: 'results',
};

// ===========================
// 完了理由
// ===========================

export const eggBootTimingCompletionReasonLabels: LocaleMap<Record<EggBootTimingCompletion['reason'], string>> = {
  ja: {
    completed: '検索完了',
    stopped: 'ユーザー停止',
    'max-results': '結果上限到達',
    error: 'エラー終了',
  },
  en: {
    completed: 'Search Completed',
    stopped: 'Stopped by User',
    'max-results': 'Max Results Reached',
    error: 'Error',
  },
};

// ===========================
// Worker 設定
// ===========================

export const eggBootTimingWorkerSettingsLabels = {
  workerCount: {
    ja: 'Worker 数',
    en: 'Worker Count',
  } satisfies LocaleText,
  workerCountHint: {
    ja: '並列処理に使用するスレッド数',
    en: 'Number of threads for parallel processing',
  } satisfies LocaleText,
  autoDetect: {
    ja: '自動検出',
    en: 'Auto-detect',
  } satisfies LocaleText,
};

// ===========================
// バリデーションエラー
// ===========================

export const eggBootTimingValidationErrors: LocaleMap<Record<string, string>> = {
  ja: {
    'invalid-datetime': '開始日時が無効です',
    'range-too-large': '検索範囲は1秒から1年以内である必要があります',
    'timer0-min-max': 'Timer0の最小値は最大値以下である必要があります',
    'timer0-range': 'Timer0は0x0000-0xFFFFの範囲である必要があります',
    'vcount-min-max': 'VCountの最小値は最大値以下である必要があります',
    'vcount-range': 'VCountは0x00-0xFFの範囲である必要があります',
    'mac-address': 'MACアドレスは6バイトの配列である必要があります',
    'hour-range': '時の範囲が無効です',
    'minute-range': '分の範囲が無効です',
    'second-range': '秒の範囲が無効です',
    'user-offset': '開始advanceは0以上の整数である必要があります',
    'advance-count': '検索件数は1-1000000の範囲である必要があります',
    'max-results': '結果上限は1-100000の範囲である必要があります',
  },
  en: {
    'invalid-datetime': 'Start date/time is invalid',
    'range-too-large': 'Search range must be between 1 second and 1 year',
    'timer0-min-max': 'Timer0 min must be less than or equal to max',
    'timer0-range': 'Timer0 must be in the range 0x0000-0xFFFF',
    'vcount-min-max': 'VCount min must be less than or equal to max',
    'vcount-range': 'VCount must be in the range 0x00-0xFF',
    'mac-address': 'MAC address must be a 6-byte array',
    'hour-range': 'Hour range is invalid',
    'minute-range': 'Minute range is invalid',
    'second-range': 'Second range is invalid',
    'user-offset': 'Start advance must be a non-negative integer',
    'advance-count': 'Advance count must be in the range 1-1000000',
    'max-results': 'Max results must be in the range 1-100000',
  },
};

// ===========================
// ヘルパー関数
// ===========================

/**
 * BCP47ロケール識別子マッピング
 * NOTE: 共通ユーティリティとして src/lib/i18n/strings/common.ts への移動を検討
 */
const BCP47_BY_LOCALE: Record<SupportedLocale, string> = {
  ja: 'ja-JP',
  en: 'en-US',
};

/**
 * 数値フォーマッタ取得
 */
function getNumberFormatter(locale: SupportedLocale): Intl.NumberFormat {
  return new Intl.NumberFormat(BCP47_BY_LOCALE[locale]);
}

/**
 * 日時フォーマッタ取得
 */
function getDateTimeFormatter(locale: SupportedLocale): Intl.DateTimeFormat {
  return new Intl.DateTimeFormat(BCP47_BY_LOCALE[locale], {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

/**
 * ステータスラベル取得
 */
export function getEggBootTimingStatusLabel(
  status: EggBootTimingSearchStatus,
  locale: SupportedLocale
): string {
  return eggBootTimingRunStatusLabels[locale][status] ?? status;
}

/**
 * 完了理由ラベル取得
 */
export function getEggBootTimingCompletionReasonLabel(
  reason: EggBootTimingCompletion['reason'],
  locale: SupportedLocale
): string {
  return eggBootTimingCompletionReasonLabels[locale][reason] ?? reason;
}

/**
 * 進捗フォーマット
 */
export function formatEggBootTimingProgress(
  current: number,
  total: number,
  locale: SupportedLocale
): string {
  const formatter = getNumberFormatter(locale);
  const pct = total > 0 ? ((current / total) * 100).toFixed(1) : '0.0';
  return `${formatter.format(current)} / ${formatter.format(total)} (${pct}%)`;
}

/**
 * 経過時間フォーマット
 */
export function formatEggBootTimingElapsedTime(
  ms: number,
  locale: SupportedLocale
): string {
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) {
    return locale === 'ja' ? `${seconds}秒` : `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  if (locale === 'ja') {
    return `${minutes}分${remainingSeconds}秒`;
  }
  return `${minutes}m ${remainingSeconds}s`;
}

/**
 * 残り時間フォーマット
 */
export function formatEggBootTimingRemainingTime(
  ms: number,
  locale: SupportedLocale
): string {
  if (ms <= 0) return '--';
  return formatEggBootTimingElapsedTime(ms, locale);
}

/**
 * 日時フォーマット
 */
export function formatEggBootTimingDatetime(
  date: Date,
  locale: SupportedLocale
): string {
  return getDateTimeFormatter(locale).format(date);
}

/**
 * 結果件数フォーマット
 * NOTE: eggBootTimingResultsCountLabel と一貫性のあるフォーマット
 */
export function formatEggBootTimingResultsCount(
  count: number,
  locale: SupportedLocale
): string {
  const formatter = getNumberFormatter(locale);
  const countStr = formatter.format(count);
  const suffix = eggBootTimingResultsCountLabel[locale];
  return `${countStr} ${suffix}`;
}

/**
 * バリデーションエラー取得
 */
export function getEggBootTimingValidationError(
  key: string,
  locale: SupportedLocale
): string {
  return eggBootTimingValidationErrors[locale][key] ?? key;
}
```

## 4. コンポーネントでの使用例

### 4.1 パネルタイトル

```tsx
import { useLocale } from '@/lib/i18n/locale-context';
import { eggBootTimingSearchPanelTitle } from '@/lib/i18n/strings/egg-boot-timing-search';

export const EggBootTimingSearchPanel: React.FC = () => {
  const locale = useLocale();
  
  return (
    <div>
      <h2>{eggBootTimingSearchPanelTitle[locale]}</h2>
    </div>
  );
};
```

### 4.2 実行制御ボタン

```tsx
import { useLocale } from '@/lib/i18n/locale-context';
import {
  eggBootTimingRunButtonLabels,
  getEggBootTimingStatusLabel,
} from '@/lib/i18n/strings/egg-boot-timing-search';

export const EggBootTimingRunCard: React.FC = () => {
  const locale = useLocale();
  const status = useEggBootTimingSearchStore((s) => s.status);
  const startSearch = useEggBootTimingSearchStore((s) => s.startSearch);
  
  const isStarting = status === 'starting';
  
  return (
    <div>
      <Button onClick={startSearch} disabled={isStarting}>
        {isStarting
          ? eggBootTimingRunButtonLabels.starting[locale]
          : eggBootTimingRunButtonLabels.start[locale]}
      </Button>
      <span>
        {getEggBootTimingStatusLabel(status, locale)}
      </span>
    </div>
  );
};
```

### 4.3 進捗表示

```tsx
import { useLocale } from '@/lib/i18n/locale-context';
import {
  eggBootTimingProgressLabels,
  formatEggBootTimingProgress,
  formatEggBootTimingElapsedTime,
  formatEggBootTimingRemainingTime,
} from '@/lib/i18n/strings/egg-boot-timing-search';

export const EggBootTimingProgressDisplay: React.FC = () => {
  const locale = useLocale();
  const progress = useEggBootTimingSearchStore((s) => s.progress);
  
  if (!progress) return null;
  
  return (
    <div>
      <div>
        <span>{eggBootTimingProgressLabels.progress[locale]}:</span>
        <span>
          {formatEggBootTimingProgress(
            progress.totalCurrentStep,
            progress.totalSteps,
            locale
          )}
        </span>
      </div>
      <div>
        <span>{eggBootTimingProgressLabels.elapsedTime[locale]}:</span>
        <span>
          {formatEggBootTimingElapsedTime(progress.totalElapsedTime, locale)}
        </span>
      </div>
      <div>
        <span>{eggBootTimingProgressLabels.remainingTime[locale]}:</span>
        <span>
          {formatEggBootTimingRemainingTime(
            progress.totalEstimatedTimeRemaining,
            locale
          )}
        </span>
      </div>
      <div>
        <span>{eggBootTimingProgressLabels.found[locale]}:</span>
        <span>{progress.totalMatchesFound}</span>
      </div>
    </div>
  );
};
```

### 4.4 結果テーブル

```tsx
import { useLocale } from '@/lib/i18n/locale-context';
import {
  eggBootTimingResultsColumnLabels,
  eggBootTimingResultsEmptyMessage,
  formatEggBootTimingDatetime,
  formatEggBootTimingResultsCount,
} from '@/lib/i18n/strings/egg-boot-timing-search';

export const EggBootTimingResultsTable: React.FC = () => {
  const locale = useLocale();
  const results = useEggBootTimingSearchStore((s) => s.results);
  
  if (results.length === 0) {
    return <p>{eggBootTimingResultsEmptyMessage[locale]}</p>;
  }
  
  return (
    <div>
      <p>{formatEggBootTimingResultsCount(results.length, locale)}</p>
      <table>
        <thead>
          <tr>
            <th>{eggBootTimingResultsColumnLabels.bootDatetime[locale]}</th>
            <th>{eggBootTimingResultsColumnLabels.timer0[locale]}</th>
            <th>{eggBootTimingResultsColumnLabels.ivs[locale]}</th>
            <th>{eggBootTimingResultsColumnLabels.nature[locale]}</th>
            {/* ... */}
          </tr>
        </thead>
        <tbody>
          {results.map((result, i) => (
            <tr key={i}>
              <td>{formatEggBootTimingDatetime(result.boot.datetime, locale)}</td>
              <td>{result.boot.timer0.toString(16).toUpperCase()}</td>
              {/* ... */}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
```

## 5. 既存の共通文字列との連携

### 5.1 再利用可能な文字列

以下の既存文字列は Search(Egg) パネルでも再利用する:

| カテゴリ | ファイル | 再利用項目 |
|----------|----------|------------|
| 性格名 | `display-common.ts` | `NATURE_NAMES` |
| 特性表示 | `display-common.ts` | `ABILITY_LABELS` |
| めざパ名 | `hidden-power.ts` | `HIDDEN_POWER_TYPE_NAMES` |
| 個体値 | `individual-values.ts` | `IV_STAT_LABELS` |
| プロファイル | `profile-*.ts` | ROM/Hardware選択肢 |

### 5.2 使用例

```tsx
import { NATURE_NAMES } from '@/lib/i18n/strings/display-common';
import { HIDDEN_POWER_TYPE_NAMES } from '@/lib/i18n/strings/hidden-power';

// 性格表示
const natureName = NATURE_NAMES[locale][result.egg.egg.nature];

// めざパタイプ表示
const hpTypeName = result.egg.egg.hiddenPower.type === 'known'
  ? HIDDEN_POWER_TYPE_NAMES[locale][result.egg.egg.hiddenPower.hpType]
  : '--';
```

## 6. エクスポート設定

```typescript
// src/lib/i18n/strings/index.ts に追加

export * from './egg-boot-timing-search';
```

## 7. 参考ドキュメント

- `src/lib/i18n/locale-context.tsx` - ロケールContext
- `src/lib/i18n/strings/types.ts` - 型定義
- `src/lib/i18n/strings/egg-run.ts` - タマゴ生成の国際化例
- `src/lib/i18n/strings/search-control.ts` - 検索制御の国際化例
- `/spec/agent/pr_egg_boot_timing_search/STATE_MANAGEMENT.md` - 状態管理設計

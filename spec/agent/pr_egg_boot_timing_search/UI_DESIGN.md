# UI/UX デザイン仕様

## 1. 概要

Search(Egg) パネルのUIデザイン仕様を定義する。既存のSearchパネル、EggパネルのUIパターンを踏襲し、レスポンシブ対応を実現する。

## 2. レスポンシブレイアウト

### 2.1 デスクトップ版（3カラム）

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        EggBootTimingSearchPanel                               │
├──────────────────────┬─────────────────────┬─────────────────────────────────┤
│      Column 1        │      Column 2       │           Column 3              │
│ ┌──────────────────┐ │ ┌─────────────────┐ │ ┌───────────────────────────┐   │
│ │   ProfileCard    │ │ │  EggParamsCard  │ │ │    EggBootTimingRunCard   │   │
│ │  (プロファイル)   │ │ │  (生成条件)     │ │ │  (検索制御・進捗表示)      │   │
│ └──────────────────┘ │ └─────────────────┘ │ └───────────────────────────┘   │
│ ┌──────────────────┐ │ ┌─────────────────┐ │ ┌───────────────────────────┐   │
│ │ BootTimingParams │ │ │ EggFilterCard   │ │ │  EggBootTimingResultsCard │   │
│ │ (起動時間設定)    │ │ │ (フィルター)    │ │ │  (結果テーブル)            │   │
│ │ - 日時範囲        │ │ └─────────────────┘ │ │  - 仮想化スクロール        │   │
│ │ - Timer0/VCount  │ │                     │ │  - ソート機能              │   │
│ │ - キー入力        │ │                     │ │  - 詳細ダイアログ          │   │
│ │ - 時刻フィルター  │ │                     │ │                            │   │
│ └──────────────────┘ │                     │ └───────────────────────────┘   │
└──────────────────────┴─────────────────────┴─────────────────────────────────┘
```

### 2.2 モバイル版（スタック）

```
┌─────────────────────────────────────────┐
│        EggBootTimingSearchPanel         │
├─────────────────────────────────────────┤
│ ┌─────────────────────────────────────┐ │
│ │           ProfileCard               │ │
│ └─────────────────────────────────────┘ │
│ ┌─────────────────────────────────────┐ │
│ │        BootTimingParamsCard         │ │
│ └─────────────────────────────────────┘ │
│ ┌─────────────────────────────────────┐ │
│ │          EggParamsCard              │ │
│ └─────────────────────────────────────┘ │
│ ┌─────────────────────────────────────┐ │
│ │          EggFilterCard              │ │
│ └─────────────────────────────────────┘ │
│ ┌─────────────────────────────────────┐ │
│ │       EggBootTimingRunCard          │ │
│ │  - 検索開始/停止ボタン              │ │
│ │  - 進捗バー                         │ │
│ └─────────────────────────────────────┘ │
│ ┌─────────────────────────────────────┐ │
│ │     EggBootTimingResultsCard        │ │
│ │  - max-height: 96 (24rem)           │ │
│ │  - スクロール表示                   │ │
│ └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

### 2.3 レスポンシブ判定

```typescript
import { useResponsiveLayout } from '@/hooks/use-mobile';

export const EggBootTimingSearchPanel: React.FC = () => {
  const { isStack, isMobile } = useResponsiveLayout();
  
  return (
    <div className={isStack ? 'flex flex-col gap-4' : 'grid grid-cols-3 gap-4'}>
      {/* レイアウト分岐 */}
    </div>
  );
};
```

## 3. 結果テーブル設計

### 3.1 テーブルカラム

既存の `ResultsCard` および `EggResultsCard` との平仄を揃える。

| カラム | 表示名(ja) | 表示名(en) | 型 | ソート | 説明 |
|--------|-----------|-----------|-----|--------|------|
| action | 操作 | Action | - | - | 詳細表示ボタン |
| bootDatetime | 起動日時 | Boot Time | Date | ✓ | ISO形式で表示 |
| timer0 | Timer0 | Timer0 | hex | ✓ | 0xXXXX形式 |
| vcount | VCount | VCount | hex | ✓ | 0xXX形式 |
| keyInput | キー入力 | Key Input | string | - | キー名リスト |
| lcgSeed | LCG Seed | LCG Seed | hex | - | 16桁16進数 |
| mtSeed | MT Seed | MT Seed | hex | - | LCGから導出 |
| advance | Advance | Advance | number | ✓ | 消費数 |
| nature | 性格 | Nature | string | ✓ | 性格名 |
| ability | 特性 | Ability | string | - | 特性1/2/夢 |
| gender | 性別 | Gender | string | - | ♂/♀/- |
| shiny | 色違い | Shiny | string | ✓ | ◇/★/- |
| ivs | 個体値 | IVs | string | - | H-A-B-C-D-S |
| hiddenPower | めざパ | HP | string | - | タイプ/威力 |
| stable | 安定 | Stable | string | - | ○/× (NPC消費時) |

### 3.2 カラム優先度

デスクトップ/モバイルでの表示優先度：

**常時表示（高優先度）:**
- action（詳細ボタン）
- bootDatetime
- advance
- shiny
- ivs

**デスクトップのみ表示（中優先度）:**
- timer0
- vcount
- keyInput
- lcgSeed
- mtSeed
- nature
- ability
- gender
- hiddenPower
- stable

### 3.3 テーブル実装

```typescript
// 仮想化対応テーブル
import { useTableVirtualization } from '@/hooks/use-table-virtualization';

const COLUMN_COUNT = 15;
const ROW_HEIGHT = 36;

export const EggBootTimingResultsTable: React.FC = () => {
  const virtualization = useTableVirtualization({
    rowCount: results.length,
    defaultRowHeight: ROW_HEIGHT,
    overscan: 8,
  });
  
  // ... テーブル実装
};
```

## 4. 詳細表示ダイアログ

### 4.1 ダイアログ構成

既存の `ResultDetailsDialog` パターンを踏襲：

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ ╳                        検索結果詳細 / Result Details                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────┬────────────────────────────────────┐   │
│  │ LCG Seed                        │ MT Seed                            │   │
│  │ 0x1234567890ABCDEF [Copy]       │ 0x12345678 [Copy]                  │   │
│  │ ↳ クリックでGenerationにコピー  │ ↳ クリックでクリップボードにコピー │   │
│  └─────────────────────────────────┴────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 起動日時 / Boot Date-Time                                            │   │
│  │ 2025-01-15 12:34:56                                                  │   │
│  │ ↳ クリックでBoot-Timingにコピー                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌────────────┬────────────┬────────────┬────────────┐                      │
│  │ Timer0     │ VCount     │ Frame      │ Key Input  │                      │
│  │ 0x0C79     │ 0x60       │ 8          │ L+R, Start │                      │
│  └────────────┴────────────┴────────────┴────────────┘                      │
│                                                                              │
│  ─────────────────────────────────────────────────────────                   │
│  個体情報 / Pokémon Info                                                    │
│  ─────────────────────────────────────────────────────────                   │
│                                                                              │
│  ┌────────────┬────────────┬────────────┬────────────┐                      │
│  │ Advance    │ 性格       │ 特性       │ 性別       │                      │
│  │ 1234       │ ようき     │ 夢特性     │ ♂         │                      │
│  └────────────┴────────────┴────────────┴────────────┘                      │
│                                                                              │
│  ┌────────────┬────────────┬────────────┬────────────┬────────────┬───────┐ │
│  │ HP         │ Atk        │ Def        │ SpA        │ SpD        │ Spe   │ │
│  │ 31         │ 31         │ 31         │ ?          │ 31         │ 31    │ │
│  └────────────┴────────────┴────────────┴────────────┴────────────┴───────┘ │
│                                                                              │
│  ┌────────────┬────────────┬────────────────────────────────────────────┐   │
│  │ 色違い     │ めざパ     │ PID                                        │   │
│  │ ★ (星型)  │ 竜/70      │ 0x12345678                                 │   │
│  └────────────┴────────────┴────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 安定性: ○ 安定 / Stable: Yes                                         │   │
│  │ ↳ NPC消費を考慮した場合でも、この消費は安定しています                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 ダイアログコンポーネント

```typescript
interface EggBootTimingResultDetailsDialogProps {
  result: EggBootTimingSearchResult | null;
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
}

export function EggBootTimingResultDetailsDialog({
  result,
  isOpen,
  onOpenChange,
}: EggBootTimingResultDetailsDialogProps) {
  const locale = useLocale();
  const { copySeedToGeneration, copyBootTimingToGeneration } = 
    useEggBootTimingResultClipboard(locale);

  if (!result) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-xl md:max-w-2xl lg:max-w-3xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{eggBootTimingResultDetailsTitle[locale]}</DialogTitle>
        </DialogHeader>
        
        {/* Seed情報セクション */}
        <SeedInfoSection result={result} onCopy={copySeedToGeneration} />
        
        {/* 起動時間セクション */}
        <BootTimingSection result={result} onCopy={copyBootTimingToGeneration} />
        
        {/* パラメータセクション */}
        <ParametersSection result={result} />
        
        {/* 個体情報セクション */}
        <PokemonInfoSection result={result} />
        
        {/* 安定性セクション (NPC消費考慮時) */}
        {result.isStable !== undefined && (
          <StabilitySection isStable={result.isStable} />
        )}
      </DialogContent>
    </Dialog>
  );
}
```

## 5. Export 機能設計

### 5.1 Export オプション

既存の `SearchExportButton` パターンを踏襲：

```typescript
interface EggBootTimingExportOptions {
  // 出力形式
  format: 'csv' | 'json' | 'txt';
  
  // 含めるデータ
  includeBootTiming: boolean;      // 起動時間情報
  includeEggDetails: boolean;      // 個体詳細（めざパ等）
  includeStability: boolean;       // 安定性フラグ
}
```

### 5.2 Export フォーマット

#### CSV形式

```csv
BootDatetime,Timer0,VCount,KeyInput,LcgSeed,MtSeed,Advance,Nature,Ability,Gender,Shiny,HP,Atk,Def,SpA,SpD,Spe,HiddenPower,PID,Stable
2025-01-15T12:34:56Z,0x0C79,0x60,"L+R,Start",0x1234567890ABCDEF,0x12345678,1234,Jolly,Hidden,Male,Star,31,31,31,?,31,31,Dragon/70,0x12345678,Yes
```

#### JSON形式

```json
{
  "exportedAt": "2025-01-15T12:34:56Z",
  "searchParams": {
    "startDatetime": "2025-01-15T00:00:00Z",
    "rangeSeconds": 3600,
    "timer0Range": { "min": 3193, "max": 3195 },
    "vcountRange": { "min": 96, "max": 96 }
  },
  "results": [
    {
      "boot": {
        "datetime": "2025-01-15T12:34:56Z",
        "timer0": "0x0C79",
        "vcount": "0x60",
        "keyInput": ["L", "R", "Start"]
      },
      "seeds": {
        "lcg": "0x1234567890ABCDEF",
        "mt": "0x12345678"
      },
      "egg": {
        "advance": 1234,
        "nature": "Jolly",
        "ability": 2,
        "gender": "male",
        "shiny": 2,
        "ivs": [31, 31, 31, 32, 31, 31],
        "hiddenPower": { "type": "Dragon", "power": 70 },
        "pid": "0x12345678",
        "isStable": true
      }
    }
  ]
}
```

### 5.3 Export コンポーネント

```typescript
interface EggBootTimingExportButtonProps {
  results: EggBootTimingSearchResult[];
  disabled?: boolean;
}

export function EggBootTimingExportButton({
  results,
  disabled = false,
}: EggBootTimingExportButtonProps) {
  const locale = useLocale();
  const [isOpen, setIsOpen] = useState(false);
  const [exportOptions, setExportOptions] = useState<EggBootTimingExportOptions>({
    format: 'csv',
    includeBootTiming: true,
    includeEggDetails: true,
    includeStability: true,
  });

  const handleExport = async (download: boolean = true) => {
    const content = EggBootTimingResultExporter.export(results, exportOptions, locale);
    
    if (download) {
      const filename = EggBootTimingResultExporter.generateFilename(exportOptions.format);
      EggBootTimingResultExporter.downloadFile(content, filename);
    } else {
      await EggBootTimingResultExporter.copyToClipboard(content);
    }
    
    setIsOpen(false);
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" disabled={disabled || results.length === 0}>
          <Download size={16} />
          {eggBootTimingExportTriggerLabel[locale]}
        </Button>
      </DialogTrigger>
      <DialogContent>
        {/* Export オプションUI */}
      </DialogContent>
    </Dialog>
  );
}
```

## 6. i18n文字列（UIデザイン用追加）

```typescript
// 結果詳細ダイアログ
export const eggBootTimingResultDetailsTitle: LocaleText = {
  ja: '検索結果詳細',
  en: 'Search Result Details',
};

export const eggBootTimingResultSectionLabels = {
  seedInfo: { ja: 'Seed情報', en: 'Seed Information' },
  bootTiming: { ja: '起動条件', en: 'Boot Conditions' },
  parameters: { ja: 'パラメータ', en: 'Parameters' },
  pokemonInfo: { ja: '個体情報', en: 'Pokémon Info' },
  stability: { ja: '安定性', en: 'Stability' },
};

export const eggBootTimingStabilityLabels = {
  stable: { ja: '○ 安定', en: '○ Stable' },
  unstable: { ja: '× 不安定', en: '× Unstable' },
  stableHint: {
    ja: 'NPC消費を考慮した場合でも、この消費は安定しています',
    en: 'This advance is stable even when considering NPC consumption',
  },
  unstableHint: {
    ja: 'NPC消費により、この消費は不安定になる可能性があります',
    en: 'This advance may be unstable due to NPC consumption',
  },
};

// Export関連
export const eggBootTimingExportLabels = {
  triggerLabel: { ja: 'エクスポート', en: 'Export' },
  dialogTitle: { ja: '検索結果のエクスポート', en: 'Export Search Results' },
  includeBootTiming: { ja: '起動時間情報を含める', en: 'Include boot timing info' },
  includeEggDetails: { ja: '個体詳細を含める', en: 'Include egg details' },
  includeStability: { ja: '安定性フラグを含める', en: 'Include stability flag' },
};
```

## 7. アクセシビリティ

### 7.1 ARIA属性

```tsx
<PanelCard
  role="region"
  aria-labelledby="egg-boot-timing-results-title"
>
  <h2 id="egg-boot-timing-results-title">
    {eggBootTimingResultsPanelTitle[locale]}
  </h2>
  <table role="grid" aria-label={eggBootTimingResultsTableAriaLabel[locale]}>
    {/* ... */}
  </table>
</PanelCard>
```

### 7.2 キーボードナビゲーション

- Tab: テーブル行間の移動
- Enter: 詳細ダイアログを開く
- Escape: ダイアログを閉じる
- Arrow Keys: ソート順変更（ヘッダー列）

## 8. 参考ドキュメント

- `src/components/search/results/ResultsCard.tsx` - 既存検索結果テーブル
- `src/components/search/results/ResultDetailsDialog.tsx` - 既存詳細ダイアログ
- `src/components/egg/EggResultsCard.tsx` - 既存タマゴ結果テーブル
- `src/components/search/results/SearchExportButton.tsx` - 既存Export機能
- `src/lib/export/result-exporter.ts` - 既存Export実装

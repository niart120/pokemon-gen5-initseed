# ポケモン生成機能 仕様書 (MVP適合版)

## 1. 概要

### 1.1 目的
初期Seed (64bit) から連続する乱数列を列挙し、BW/BW2 におけるエンカウントポケモンの核心乱数属性 (PID, 性格, 特性スロット, 色違い, 同期適用, スロット値 等) を高速ストリーミング表示する。

### 1.2 対象ゲーム
- ポケットモンスターブラック・ホワイト（第5世代）
- ポケットモンスターブラック2・ホワイト2（第5世代）

### 1.3 対象エンカウント方法 (MVP)
- 野生 / Surf / Fishing / ShakingGrass / DustCloud / Shadow / Bubble 系列 (encounterType enum 値に準拠)
- 固定シンボル / 個体値生成 完全計算は FUTURE (未実装)

## 2. 機能要件

### 2.1 入力条件

#### 2.1.1 基本パラメータ
- **初期Seed値**: 64bit値（16進数表記）
- **エンカウント方法**: 野生/固定シンボル/つり
- **シンクロ設定**: シンクロ有効/無効、対象性格
- **表ID**: トレーナーID（16bit、0-65535）
- **裏ID**: 秘密ID（16bit、0-65535）

#### 2.1.2 エンカウント方法別パラメータ (MVP範囲外)
位置/レベル/種族直接指定は現段階 UI / Worker では未使用。Encounter Slot 解決は FUTURE。

### 2.2 出力情報

#### 2.2.1 基本ポケモン情報 (MVP 表示フィールド)
- advance (消費乱数インデックス)
- seed (hex/dec)
- pid (hex/dec)
- nature (index → UI で名称)
- ability_slot (0/1)
- encounter_slot_value
- encounter_type
- shiny_type (0:通常 1:Square 2:Star)
- sync_applied (boolean)

#### 2.2.2 乱数関連情報 (FUTURE)
- 性別・個体値 (IV)・実乱数列表示は未実装

#### 2.2.3 追加情報
- シンクロ適用フラグ (実装済)
- レポ針等のメタ情報: FUTURE

### 2.3 表示機能 (現状)
- テーブル: 生成順 (advance 昇順)
- ハイライト: 色違い (shiny_type !=0)
- ソート/高度フィルタ: FUTURE
- 詳細パネル: FUTURE

## 3. 技術仕様

### 3.1 データ構造

#### 3.1.1 入力パラメータ (Worker GenerationParams 要約)
| Param | 説明 | 制約 |
|-------|------|------|
| baseSeed | 初期Seed bigint | 0 ≤ < 2^64 |
| offset | 開始オフセット | 0..maxAdvances |
| maxAdvances | 消費上限 | 1..1,000,000 |
| maxResults | 収集上限 | 1..100,000 且つ ≤ maxAdvances |
| version | ゲーム版 | enum |
| encounterType | エンカウント種別 | enum |
| tid / sid | 表/裏ID | 0..65535 |
| syncEnabled | シンクロ有効 | boolean |
| syncNatureId | 性格ID | 0..24 |
| stopAtFirstShiny | 最初の色違い停止 | boolean |
| stopOnCap | maxResults 到達で停止 | boolean (default true) |
| progressIntervalMs | 進捗間隔 | default 250 |
| batchSize | バッチ生成数 | 1..10,000 |

#### 3.1.2 出力データ (RawLike)
```ts
type RawLike = {
  seed: bigint; pid: number; nature: number; ability_slot: number; gender_value: number;
  encounter_slot_value: number; encounter_type: number; level_rand_value: bigint;
  shiny_type: number; sync_applied: boolean; advance: number;
};
```

### 3.2 計算アルゴリズム

#### 3.2.1 LCG（線形合同法）
ポケモンBW/BW2で使用される乱数生成式：
```
次の乱数 = (現在の乱数 × 0x5D588B656C078965 + 0x269EC3)
```
※ 64bit LCGを用い、乱数の上位32bitを各判定に使用します。

#### 3.2.2 ポケモン生成手順 (MVP 抜粋)

##### Step 1: エンカウント判定
1. 野生の場合：エンカウントスロット決定（乱数消費1回）
2. つりの場合：つり成功判定 + スロット決定
3. 固定の場合：スキップ

##### Step 2: ポケモン基本情報生成
1. **性格決定**（乱数消費1回）
   - シンクロ有効時：50%で指定性格、50%でランダム
   - シンクロ無効時：25種類からランダム選択
   
2. **特性決定**（乱数消費1回）
   - 通常特性：0-1で特性1/2決定
   - 隠れ特性：エンカウント方法により判定

##### Step 3: 個体値生成 (FUTURE)
IV 計算は現段階未公開 (RawLike に含まず)

##### Step 4: 色違い・性別判定
1. **PID生成**（乱数消費2回）
   - 上位16bit、下位16bitを別々に生成
   
2. **色違い判定**
   ```
   shinyValue = (trainerId ^ secretId ^ pidHigh ^ pidLow) < 8
   ```
   
3. **性別判定**
   - 種族の性別比率に基づいて判定

### 3.3 実装方針 (現状)

#### 3.3.1 計算エンジン
- **WebAssembly（Rust）**: 高速な乱数生成・計算処理（全ての計算ロジックはWASM側で実装）
- **TypeScript**: UI・データ管理・表示処理（計算処理は行わず、WASMの結果をパース・表示のみ担当）

#### 3.3.2 ポケモンデータ
種族/エンカウントテーブルは段階的導入。現行は encounter_slot_value をそのまま表示。

#### 3.3.3 パフォーマンス考慮
- **バッチ処理**: 大量生成時の効率化
- **プログレス表示**: 長時間計算の進捗表示
- **結果キャッシュ**: 同条件での再計算回避

## 4. UI設計 (現状要約)

### 4.1 入力画面
- **シンプルなフォーム**: 必要最小限の入力項目
- **プリセット機能**: よく使用する設定の保存
- **バリデーション**: 入力値の検証・エラー表示

### 4.2 結果画面
- **表形式表示**: スプレッドシート風の見やすい表示
- **色分け**: 色違い、理想個体値等の強調表示
- **エクスポート**: CSV、JSON形式での出力

### 4.3 設定画面
- **詳細設定**: 上級者向けオプション
- **データ管理**: ポケモンデータの更新・管理

## 5. データ要件 (簡略)

### 5.1 ポケモン種族データ
```typescript
interface PokemonSpecies {
  id: number;                   // 全国図鑑番号
  name: string;                 // ポケモン名
  abilities: string[];          // 通常特性1, 2, 隠れ特性
  genderRatio: number;          // 性別比率（-1:性別不明, 0-254）
}
```

### 5.2 エンカウントテーブル (FUTURE)
```typescript
interface EncounterTable {
  location: string;             // 場所名
  encounterType: EncounterType; // エンカウント方法
  slots: EncounterSlot[];       // エンカウントスロット情報
}

interface EncounterSlot {
  pokemon: string;              // ポケモン種族名
  probability: number;          // 出現確率（%）
  levelRange: {
    min: number;
    max: number;
  };
  conditions?: {                // 出現条件
    timeOfDay?: string[];
    season?: string[];
    special?: string;           // 特殊条件
  };
}
```

## 6. 拡張性考慮 / Roadmap

### 6.1 将来対応
- **新機能追加**: めざめるパワー等

### 6.2 モジュール設計
- **計算エンジン分離**: 他プロジェクトでの再利用
- **データ層分離**: ポケモンデータの独立管理
- **UI層分離**: 表示形式の柔軟な変更

## 7. テスト要件 (実施状況)

### 7.1 単体テスト
- RNG / Shiny 判定 Rust 単体テスト済
- 性格 / 同期適用 パス検証テスト済

### 7.2 統合テスト
- Node 環境 Worker 非対応時: ガードで安全成功 (D1)
- 早期終了シナリオ (max-advances / max-results / first-shiny) カバレッジ

### 7.3 受け入れテスト
- E2E (Playwright-MCP) 追加予定

## 8. Worker プロトコル要約
Messages: START_GENERATION / PROGRESS / RESULT_BATCH / COMPLETE / STOPPED / ERROR (+ READY, PAUSE/RESUME 予備)
Completion Reasons: max-advances | max-results | first-shiny | stopped | error
Validation 主項目: baseSeed, maxAdvances, maxResults ≤ maxAdvances, batchSize, syncNatureId, offset

## 9. 参考資料

### 8.1 技術資料
- [ポケモン第5世代乱数調整](https://rusted-coil.sakura.ne.jp/pokemon/ran/ran_5.htm) : 乱数調整の基礎
- [BWなみのり、つり、大量発生野生乱数](https://xxsakixx.com/archives/53402929.html) :
  なみのりやつり、大量発生の個体生成
- [BW出現スロットの閾値](https://xxsakixx.com/archives/53962575.html) : 出現スロットの計算方法


### 8.2 データソース
- [ポケモン攻略DE.com](http://blog.game-de.com/pokedata/pokemon-data/) : ポケモン種族データ
- [ポケモンの友(B)](https://pokebook.jp/data/sp5/enc_b) : ブラックのエンカウントテーブル
- [ポケモンの友(W)](https://pokebook.jp/data/sp5/enc_w) : ホワイトのエンカウントテーブル
- [ポケモンの友(B2)](https://pokebook.jp/data/sp5/enc_b2) : ブラック2のエンカウントテーブル
- [ポケモンの友(W2)](https://pokebook.jp/data/sp5/enc_w2) : ホワイト2のエンカウントテーブル

---

**作成日**: 2025年8月2日  
**バージョン**: 1.1  
**作成者**: GitHub Copilot  
**更新日**: 2025年8月12日  
**レビュー状況**: MVP 反映

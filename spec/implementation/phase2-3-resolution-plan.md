# Phase 2-3: Species / Ability / Gender 解決リファクタ計画

目的
- エンカウントテーブル・種族・特性・性別判定の実装を一元化し、mainブランチのJSONエンカウント仕様に整合。
- データは外部API由来のローカルJSONを生成して参照（再現可能なデータパイプライン）。

方針（決定事項）
1) エンカウントは既存のJSONテーブル群（src/data/encounters/**）を唯一のソースとする。
   - PR30 の簡易BASIC_ENCOUNTER_TABLEと独自Resolver（src/lib/integration/resolvers.ts）は廃止。
2) 性別判定は「female閾値（0-255）基準」に統一。
   - gender_value < female_threshold → Female、その以外 → Male。
   - Genderless/固定性別はデータで明示。
   - 共通ユーティリティ化して単一実装を使用。
3) 特性解決は ability_slot による通常特性1/2の選択のみに限定。
   - 隠れ特性は別フラグ（例: isHiddenAbility）でのみ選択。slotからの暗黙フォールバックは行わない。
   - WASM出力契約に応じてRawPokemonDataへフラグ追加の検討（必要に応じて別PR化）。
4) データソースは外部API（PokéAPI）を利用し、生成したJSONを参照。
   - National Dex 495-649（Gen5）について、性別率・特性（hidden含む）を取得。
   - 取得スクリプトは Node 18+ 標準 fetch を使用。生成物はリポジトリにコミット。

生成データのスキーマ（案）
- 保存先: src/data/species/generated/gen5-species.json
- 例:
  {
    "495": {
      "nationalDex": 495,
      "name": "Snivy",
      "gender": { "type": "ratio", "femaleThreshold": 31 },
      "abilities": { "ability1": "Overgrow", "ability2": null, "hidden": "Contrary" }
    },
    ...
  }
- gender: { type: 'genderless' } | { type: 'fixed', fixed: 'male' | 'female' } | { type: 'ratio', femaleThreshold: number }

段階的作業計画（各段で go/no go 確認）
Phase 1: データパイプライン整備（コード追加のみ、機能切替は未実施）
- scripts/fetch-gen5-species.ts を追加（PokéAPI→gen5-species.json を生成）。
- 型: src/types/species.ts（gender/abilitiesのスキーマを定義）。
- npm scripts 追加: "fetch:species:gen5"。
- 生成物はコミット対象（安定化のため）。

更新: Phase 1-ext（データ生成の拡張）
- 範囲: National Dex 1–649（第5世代まで）
- 収集フィールド:
  - names: { en, ja }
  - gender: { type, femaleThreshold|fixed }
  - baseStats: { hp, attack, defense, specialAttack, specialDefense, speed }
  - abilities: { ability1, ability2?, hidden? }（各 { key, names: { en, ja } }）
  - heldItems: { black: ItemEntry[], white: ItemEntry[], black-2: ItemEntry[], white-2: ItemEntry[] }
    - ItemEntry: { key, names: { en, ja }, rarity }
- スクリプト: scripts/fetch-gen5-species.js（収集＋ローカライズ）
- 出力: src/data/species/generated/gen5-species.json
- 既存エイリアス: PokéAPI準拠へ統合。不要になれば削除。

Phase 2: サービス統合と重複排除
- PokemonIntegrationService で生成JSONを参照するローダを追加。
- 性別判定ユーティリティ（female閾値基準）を src/lib/services 直下に実装し、統一利用。
- PR30で追加された src/lib/integration/resolvers.ts と src/data/species/encounter-tables.ts / core-species.ts を撤去。
- ability解決は ability_slot→ability1/2 のみ。hiddenはフラグに限定（フラグ導入はPhase 2で実施可/別PRでも可）。

Phase 3: テスト更新
- 境界テストを femaleThreshold 基準に修正（例: 12.5% Female→閾値31 等）。
- IntegrationService 経由のE2E/統合テストに合わせてテストを再配置。
- 隠れ特性はフラグが立つケースに限定して検証。

Phase 4: クリーニング
- Lint修正（未使用変数・React Hooksルール等の警告/エラー解消）。
- ドキュメント整備（spec/implementation/phase2-api.md の追記、README抜粋更新）。

リスクと対処
- 外部API変更/一時不通 → 生成物をリポジトリに保持、再取得は手動。
- 既存型との齟齬 → src/types で統一型を定義し、旧 raw-pokemon-data.ts は `src/types/pokemon-enhanced.ts` に統合済み。
- 隠れ特性判定 → 現行はslotに依存しない。WASMの仕様確認後にフラグ導入を検討。

影響範囲
- データ参照先の切替により、種族/性別/特性の一部結果が修正される可能性。
- 独自Resolver撤去に伴い、PR30で追加の一部テストは再設計。

承認ポイント
- Phase 1 完了後: 生成スクリプト・スキーマ・JSON出力の構造確認。
- Phase 2 完了後: IntegrationServiceの解決結果が現行仕様と一致/改善しているか確認。
- Phase 3 完了後: 全テストパス、境界条件の妥当性確認。

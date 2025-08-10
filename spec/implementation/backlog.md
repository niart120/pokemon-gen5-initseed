# Backlog

- UI: Nature 表示ローカライズ
  - 概要: DomainNatureNames（英語）を基準に、UI で ja 等の表示マップを適用する
  - 受け入れ基準:
    - `getLocalizedNatureName(id: number, locale: string)` の提供
    - 既存 UI コンポーネント（表示箇所）でローカライズ関数を使用
    - 既存テストは英語のまま維持、UI 表示のスナップショットまたは単体テストを追加
  - 留意点:
    - DomainNatureNames は単一ソース（英語）として維持
    - ローカライズは UI 層限定で実施し、型やドメイン定数へ混在させない

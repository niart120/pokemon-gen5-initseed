# Webアプリ版 6Vメタモン乱数調整ガイド

- 作業日: 2025-12-01
- 参照コミット: `6bd61341af9cc14e74786200ad3bdfd1f6215552`
- 対象タイトル: ポケットモンスター ブラック/ホワイト (BW1)

## 導入と前提
本ガイドは https://milk4724.hatenablog.com/entry/20231230/00000000 の内容をもとに、本リポジトリの Web アプリで 6V メタモン (ジャイアントホール B1F) を狙う手順を整理した。乱数調整未経験者でも追えるよう、アプリ内の `SearchPanel` で初期Seedを探し、`GenerationPanel` でログと検証を行う構成に置き換えている。

### 用語のサマリ
- `初期Seed`: BW 起動直後に決定される64bit値。`SearchPanel` の検索結果一覧に `seed` として表示される。
- `Timer0`: 本体ごとに揺れるハードウェア内部のカウンタ。`SearchParamsCard` で範囲指定し、結果一覧にも表示される。
- `シンクロ`: 先頭に配置したポケモンの性格を野生遭遇時に反映する特性。`GenerationPanel` のフィルタで性格確認が可能。
- `消費数`: 乱数を進めた回数。BW1 の野生6Vテンプレートは消費0/1で用途が分かれる。
- `#tool:` 記法: MCP ツールを呼び出す指示。例: `#tool:take_screenshot SearchPanel` は SearchPanel の状態を撮影して記録する際に利用する。

### ブログ手順と Web アプリの対応
| ブログ記載フェーズ | Web アプリでの操作 | 補足 |
| --- | --- | --- |
| 1. 初期Seed検索 (5genSearch) | `SearchPanel` → `SearchParamsCard` と `TargetSeedsCard` で条件入力し、`SearchControlCard` で検索実行 | 時間範囲・キー指定・Timer0 範囲を対応させる |
| 2. リスト表示・Timer0確認 | `ResultsCard` で日時/Timer0を並べ替え、`ResultDetailsDialog` で seed 情報を保存 | 必要に応じ `#tool:take_screenshot SearchPanel` |
| 3. 乱数消費・実機操作 | アプリ上では記録のみ。実機ではブログ同様に C-GEAR オフ・レポート消費を実施 | 進捗ログは `#tool:mcp_chromedevtool_list_console_messages SearchPanel` で保存可 |
| 4. 成功例・失敗時分析 | `GenerationPanel` → `GenerationResultsTableCard` で手持ちステータスを入力し検証 | スクショは `#tool:take_screenshot GenerationPanel` |

## 準備 (導入フェーズ)
1. **実機要件**
   - ニンテンドーDS / DS Lite (DSi/3DS系は不可)。
   - ソフトは `ポケットモンスター ブラック` または `ホワイト`。
   - パラメータ (MACアドレス、VCount、既知の Timer0 帯) を事前に特定済みであること。
   - ジャイアントホール B1F の霧を晴らし、徘徊ポケモンがいない状態にする。
2. **Webアプリ要件**
   - `npm run dev` または `npm run dev:agent` でローカルサーバーを起動し、ブラウザで `http://localhost:5173` を開く。
   - プロフィール (`ProfileCard`) に本体・ROM情報を保存しておくと `SearchPanel`/`GenerationPanel` 双方で再利用できる。
3. **記録習慣 (MCP)**
   - UIの状態記録: `#tool:take_screenshot SearchPanel`
   - 進捗ログ取得: `#tool:mcp_chromedevtool_list_console_messages SearchPanel`
   - 生成結果記録: `#tool:take_screenshot GenerationPanel`

## 測定 (Timer0 と前提確定)
1. **Timer0 サンプリング**
   - `TargetSeedsCard` のテンプレートから「BW 固定・野生 6V」を読み込み、Timer0 帯ごとに Seed 候補 (例: `0x14B11BA6`) を5種類確保する。
   - `SearchParamsCard` → `Timer0` 範囲を広めに設定 (例: `0xC67`〜`0xC69`)。時間範囲は100年単位に固定し、初回はブログ同様 `秒=11` 固定で開始する。
   - `SearchControlCard` で CPU 並列 (デフォルト) を選択、`Wake Lock` を有効にして測定を開始。
2. **実機での捕獲確認**
   - DS 本体の時刻を検索結果の日時に合わせ、1秒前でソフト選択 → 白画面中にキー (例: `A+B+→`) を押し続ける。
   - C-GEAR オフで再開後、`あまいかおり` で遭遇し捕獲。
   - 捕獲結果は `GenerationPanel` → `GenerationResultsTableCard` に入力し、実測 Timer0 と一致するか確認。
3. **Timer0 の頻出値決定**
   - 3回以上同じ Timer0 が出たら、その値のみを `SearchParamsCard` の範囲に残す。
   - 計測ログを `#tool:mcp_chromedevtool_list_console_messages SearchPanel` で保存しておくと再検証が容易。

## シード探索 (SearchPanel の具体操作)
1. **検索条件入力**
   1. `SearchParamsCard`
      - `開始日/終了日`: 狙いたい年代を広く設定 (例: `2000-01-01`〜`2099-12-31`)。
      - `時間範囲`: `時=19`, `分=24`, `秒=11` のようにブログ推奨値から狭めて入力。
      - `キー入力`: `GameController` ボタンから `KeyInputDialog` を開き、押しやすいキー (A/B/→など) を選択。
   2. `TargetSeedsCard`
      - 「テンプレート」ボタンで `BW 固定・野生 6V` を読み込み、シンクロに合わせて必要なら他テンプレートを追加。
      - 追加した seed をメモ (例: `0xFC4AA3AC`)。`#tool:take_screenshot SearchPanel` で設定証跡を確保。
2. **検索実行**
   - `SearchControlCard` の `開始` ボタンで検索。進捗は `SearchProgressCard` で確認し、バックグラウンド実行中でも Wake Lock で画面を保持。
   - ヒットが出たら `ResultsCard` で日時の昇順に並べ、`ResultDetailsDialog` から `seed / Timer0 / keyCode` をコピーし、`note/bw_ditto_6V/GPT-5.1-Codex` 配下にメモを残す。
   - 長時間検索になる場合は `#tool:mcp_chromedevtool_list_console_messages SearchPanel` で節目ごとにログをダンプ。

## 検証 (GenerationPanel の活用)
1. `GenerationRunCard` でシナリオ名 (例: `Ditto6V_C68`) を作成し、使用 Seed・キー入力・Timer0 を控える。
2. `GenerationParamsCard` にはシンクロ性格、消費数 (`0` or `1`)、遭遇フィールド (`草むら/洞窟`) を入力。
3. `GenerationFilterCard` で実測ステータスを入力し、結果テーブルに表示される期待値と一致するか確認。
4. 検証過程を `#tool:take_screenshot GenerationPanel` で記録し、必要に応じ `#tool:mcp_chromedevtool_list_console_messages GenerationPanel` でフィルタ条件をログ化。

## 捕獲と仕上げ (実機手順)
1. DS 時刻をヒットした日時に再設定し、1秒前でソフト選択。
2. 白画面中は `SearchPanel` で指定したキーを長押し。押しづらいキーは除外済みであることを再確認。
3. `続きから` は C-GEAR オフ。もしオンにした場合は即座にリセットして再トライ。
4. 予定した消費数だけレポート、またはペラップ確認で調整。
5. `あまいかおり` → メタモン遭遇 → 捕獲。
6. `GenerationPanel` に捕獲結果を入力し、IV/性格/Timer0 が一致すれば完了。`#tool:take_screenshot GenerationPanel` で成功ログを残す。

## サンプルシナリオ (Timer0=0xC68)
1. **検索条件**
   - Seed: `F4E462E25FD88FCC` (ブログ記載)
   - 時刻: `2008-10-02 19:24:11`
   - キー入力: `A+B+→`
   - Timer0 範囲: `0xC68` 固定
   - 消費数: `1` (出現前にレポート1回)
2. **SearchPanel 設定例**
   - `時間範囲` → `時 19~19`, `分 24~24`, `秒 11~11`
   - `TargetSeedsCard` に `0x14B11BA6` などテンプレ Seed を追加し、`SeedCalculator` の `parseTargetSeeds` が有効であることを確認。
3. **実機操作**
   - 19:24:10 でソフト選択、白画面中に `A+B+→`。
   - 再開後すぐレポート (1消費) → `あまいかおり` → 捕獲。
4. **検証**
   - 性格「いじっぱり」シンクロ成功かを `GenerationPanel` のフィルタでチェック。
   - ずれた場合は `Timer0` ズレとみなし、`SearchPanel` の結果一覧から別 Timer0 seed を再選択。

## トラブルシュート
- **ヒットが出ない**: 時刻範囲の秒数を±2広げ、`TargetSeedsCard` で Seed 種類を増やす。Timer0 範囲も `±1` で再検索。
- **Timer0 が毎回ズレる**: `SearchParamsCard` の Timer0 範囲を実測値中心に絞り、`SearchControlCard` で GPU モードを試して検索速度を上げつつデータ点を増やす。
- **シンクロが効かない**: `GenerationPanel` の結果にシンクロ性格欄が表示されているか確認し、先頭ポケモンが戦闘不能でないか現地で再確認。
- **消費数を間違える**: `GenerationPanel` の `消費数` フィールドで0/1を切り替え、期待遭遇数を見直す。必要に応じて `#tool:take_screenshot GenerationPanel` で比較履歴を残す。

## 成果物保存・更新
- 本ガイド: `note/bw_ditto_6V/GPT-5.1-Codex/guide.md`。
- 追加スクリーンショットは同ディレクトリに `images/` を作成し、本文から相対パスで参照する。ログは `logs/` に保存し、`#tool:mcp_chromedevtool_list_console_messages SearchPanel` などで取得した内容を置く。
- 設定を更新した場合は、作業日と対象コミットを追記し、本ファイルを更新する。

## 確認チェックリスト
- [ ] SearchPanel で Timer0・キー入力・時間範囲を設定し、テンプレ Seed を読み込んだか。
- [ ] MCP の `#tool:take_screenshot SearchPanel` で設定証跡を残したか。
- [ ] 実機で C-GEAR オフ・レポート消費を守ったか。
- [ ] GenerationPanel の結果と実測値が一致し、スクリーンショットを保存したか。
- [ ] ログ・スクショを `note/bw_ditto_6V/GPT-5.1-Codex` 配下に整理したか。

## 参照資料
- 外部: https://milk4724.hatenablog.com/entry/20231230/00000000 (6Vメタモン乱数調整 5genSearch)
- リポジトリ内:
  - `src/components/layout/SearchPanel.tsx`
  - `src/components/layout/GenerationPanel.tsx`
  - `src/components/search/configuration/SearchParamsCard.tsx`
  - `src/components/search/configuration/TargetSeedsCard.tsx`
  - `src/data/seed-templates.ts`

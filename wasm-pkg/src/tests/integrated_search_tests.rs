/// 統合Seed探索のテストコード
use crate::integrated_search::SearchResult;

#[cfg(target_arch = "wasm32")]
use crate::integrated_search::IntegratedSeedSearcher;

// WASM環境でのテスト設定
#[cfg(target_arch = "wasm32")]
use crate::sha1::{calculate_pokemon_seed_from_hash, calculate_pokemon_sha1, swap_bytes_32};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_browser);

// WASM環境用テスト
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests {
    use super::*;

    // ==== SearchResult のテスト ====

    #[wasm_bindgen_test]
    fn test_search_result() {
        let result = SearchResult::new(
            0x12345678,
            "abcdef1234567890abcdef1234567890abcdef12".to_string(),
            2012,
            6,
            15,
            10,
            30,
            45,
            0x2FFF,
            1120,
            50,
        );
        assert_eq!(result.seed(), 0x12345678);
        assert_eq!(result.hash(), "abcdef1234567890abcdef1234567890abcdef12");
        assert_eq!(result.year(), 2012);
        assert_eq!(result.month(), 6);
        assert_eq!(result.date(), 15);
        assert_eq!(result.hour(), 10);
        assert_eq!(result.minute(), 30);
        assert_eq!(result.second(), 45);
        assert_eq!(result.key_code(), 0x2FFF);
        assert_eq!(result.timer0(), 1120);
        assert_eq!(result.vcount(), 50);
    }

    // ==== IntegratedSeedSearcher のテスト ====

    #[wasm_bindgen_test]
    fn test_integrated_searcher_creation() {
        let mac = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC];
        let nazo = [0x02215f10, 0x01000000, 0xc0000000, 0x00007fff, 0x00000000];

        let searcher = IntegratedSeedSearcher::new(&mac, &nazo, "DS", 5, 8, 0, 23, 0, 59, 0, 59);
        assert!(searcher.is_ok());
    }

    #[wasm_bindgen_test]
    fn test_invalid_mac_length() {
        let mac = [0x12, 0x34, 0x56, 0x78, 0x9A]; // 5 bytes instead of 6
        let nazo = [0x02215f10, 0x01000000, 0xc0000000, 0x00007fff, 0x00000000];

        let result = IntegratedSeedSearcher::new(&mac, &nazo, "DS", 5, 8, 0, 23, 0, 59, 0, 59);
        assert!(result.is_err());
    }

    #[wasm_bindgen_test]
    fn test_invalid_nazo_length() {
        let mac = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC];
        let nazo = [0x02215f10, 0x01000000, 0xc0000000, 0x00007fff]; // 4 elements instead of 5

        let result = IntegratedSeedSearcher::new(&mac, &nazo, "DS", 5, 8, 0, 23, 0, 59, 0, 59);
        assert!(result.is_err());
    }

    #[wasm_bindgen_test]
    fn test_invalid_hardware() {
        let mac = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC];
        let nazo = [0x02215f10, 0x01000000, 0xc0000000, 0x00007fff, 0x00000000];

        let result = IntegratedSeedSearcher::new(&mac, &nazo, "INVALID", 5, 8, 0, 23, 0, 59, 0, 59);
        assert!(result.is_err());
    }

    // ==== パフォーマンステスト ====

    #[wasm_bindgen_test]
    fn test_performance_sha1_calculation() {
        use crate::sha1::{calculate_pokemon_seed_from_hash, calculate_pokemon_sha1};
        use js_sys::Date;

        web_sys::console::log_1(&"=== SHA-1計算パフォーマンステスト開始 ===".into());

        // テスト用メッセージ（実際のポケモンメッセージ形式）
        let test_message: [u32; 16] = [
            0x02215f10, 0x01000000, 0xc0000000, 0x00007fff, 0x00000000, // nazo部分
            0x12345678, 0x9ABCDEF0, 0x34561234, // MAC部分（サンプル）
            0x0C0F0F04, 0x00120000, // 日時部分（サンプル）
            0x00000000, 0x00000000, 0x00000005, // 固定値
            0x80000000, 0x00000000, 0x000001A0, // SHA-1パディング
        ];

        // 大量のSHA-1計算パフォーマンステスト
        let iterations = 100_000; // 元の規模に戻す
        let output = format!("{}回のSHA-1計算を実行します...", iterations);
        web_sys::console::log_1(&output.into());

        let start = Date::now();
        let mut total_seeds = 0u64;

        for i in 0u32..iterations {
            // 各イテレーションでメッセージを少し変更（Timer0/VCountをシミュレート）
            let mut message = test_message;
            message[8] = message[8].wrapping_add(i % 0x10000u32); // Timer0相当
            message[9] = message[9].wrapping_add(i % 263u32); // VCount相当

            // SHA-1計算
            let (h0, h1, _h2, _h3, _h4) = calculate_pokemon_sha1(&message);
            let seed = calculate_pokemon_seed_from_hash(h0, h1);
            total_seeds = total_seeds.wrapping_add(seed as u64);
        }

        let end = Date::now();
        let duration_ms = end - start;

        // 結果出力
        web_sys::console::log_1(&"=== SHA-1計算パフォーマンス結果 ===".into());
        let output = format!("計算回数: {}", iterations);
        web_sys::console::log_1(&output.into());
        let output = format!("実行時間: {:.2}ms", duration_ms);
        web_sys::console::log_1(&output.into());
        let calc_per_sec = (iterations as f64) / (duration_ms / 1000.0);
        let output = format!("1秒あたりの計算数: {:.2} calculations/sec", calc_per_sec);
        web_sys::console::log_1(&output.into());
        let avg_time_ns = (duration_ms * 1_000_000.0) / iterations as f64;
        let output = format!("1回あたりの平均時間: {:.2} ns", avg_time_ns);
        web_sys::console::log_1(&output.into());
        let output = format!("チェックサム: 0x{:016X}", total_seeds);
        web_sys::console::log_1(&output.into());

        // パフォーマンス基準チェック（実性能の50%程度を基準とする）
        assert!(
            calc_per_sec > 350_000.0,
            "SHA-1計算性能が基準を下回りました: {:.2} calc/sec",
            calc_per_sec
        );

        web_sys::console::log_1(&"=== SHA-1計算パフォーマンステスト完了 ===".into());
    }

    #[wasm_bindgen_test]
    fn test_performance_datetime_lookup_comparison() {
        use crate::datetime_codes::{DateCodeGenerator, TimeCodeGenerator};
        use js_sys::Date;

        web_sys::console::log_1(&"=== 日時ルックアップ比較テスト開始 ===".into());

        let iterations = 100_000; // 元の規模に戻す

        // 1. 境界チェック付きルックアップ
        let start = Date::now();
        let mut total_codes = 0u64;

        for i in 0u32..iterations {
            let time_index = i % 86400;
            let date_index = i % 36525;

            let date_code = DateCodeGenerator::get_date_code(date_index);
            let time_code = TimeCodeGenerator::get_time_code(time_index);

            total_codes = total_codes.wrapping_add(date_code as u64 + time_code as u64);
        }

        let end = Date::now();
        let safe_duration = end - start;

        // 2. 境界チェックなしルックアップ
        let start = Date::now();
        let mut total_codes_unsafe = 0u64;

        for i in 0u32..iterations {
            let time_index = (i % 86400) as usize;
            let date_index = (i % 36525) as usize;

            let date_code = unsafe { *DateCodeGenerator::DATE_CODES.get_unchecked(date_index) };
            let time_code = unsafe { *TimeCodeGenerator::TIME_CODES.get_unchecked(time_index) };

            total_codes_unsafe =
                total_codes_unsafe.wrapping_add(date_code as u64 + time_code as u64);
        }

        let end = Date::now();
        let unsafe_duration = end - start;

        // 結果比較
        web_sys::console::log_1(&"=== 日時ルックアップ比較結果 ===".into());
        let output = format!(
            "境界チェック付き: {:.2}ms ({:.2} ns/回)",
            safe_duration,
            (safe_duration * 1_000_000.0) / iterations as f64
        );
        web_sys::console::log_1(&output.into());
        let output = format!(
            "境界チェックなし: {:.2}ms ({:.2} ns/回)",
            unsafe_duration,
            (unsafe_duration * 1_000_000.0) / iterations as f64
        );
        web_sys::console::log_1(&output.into());
        let speedup = safe_duration / unsafe_duration;
        let output = format!("性能向上: {:.1}倍", speedup);
        web_sys::console::log_1(&output.into());

        // チェックサムが同じことを確認
        assert_eq!(
            total_codes, total_codes_unsafe,
            "境界チェック有無で結果が異なる"
        );

        web_sys::console::log_1(&"=== 日時ルックアップ比較テスト完了 ===".into());
    }

    #[wasm_bindgen_test]
    fn test_performance_integrated_search() {
        use crate::datetime_codes::{DateCodeGenerator, TimeCodeGenerator};
        use js_sys::Date;

        web_sys::console::log_1(&"=== 統合Seed探索パフォーマンステスト開始 ===".into());

        // テスト用パラメータ
        let mac = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC];
        let nazo = [0x02215f10, 0x01000000, 0xc0000000, 0x00007fff, 0x00000000];
        let target_seeds = vec![0x12345678, 0x87654321, 0xABCDEF01, 0x11111111];
        let hardware = "DS";

        // 基本メッセージテンプレートを事前構築（統合探索器と同じロジック）
        let mut base_message = [0u32; 16];

        // data[0-4]: Nazo values
        for i in 0..5 {
            base_message[i] = swap_bytes_32(nazo[i]);
        }

        // MAC address setup
        let mac_lower = ((mac[4] as u32) << 8) | (mac[5] as u32);
        base_message[6] = mac_lower;
        let mac_upper = (mac[0] as u32)
            | ((mac[1] as u32) << 8)
            | ((mac[2] as u32) << 16)
            | ((mac[3] as u32) << 24);
        let gx_stat = 0x06000000u32;
        let frame = 8u32;
        let data7 = mac_upper ^ gx_stat ^ frame;
        base_message[7] = swap_bytes_32(data7);

        base_message[10] = 0x00000000;
        base_message[11] = 0x00000000;
        base_message[12] = swap_bytes_32(5); // key_input
        base_message[13] = 0x80000000;
        base_message[14] = 0x00000000;
        base_message[15] = 0x000001A0;

        // 探索範囲設定（2日間のテスト）
        let range_seconds = 2 * 24 * 3600; // 2日間
        let timer0_range = 6; // 実用的なTimer0範囲
        let vcount_range = 2; // 実用的なVCount範囲

        let total_calculations =
            range_seconds as u64 * timer0_range * vcount_range * target_seeds.len() as u64;
        let output = format!(
            "探索範囲: {}秒 × Timer0({}) × VCount({}) × 目標({}) = {} 計算",
            range_seconds,
            timer0_range,
            vcount_range,
            target_seeds.len(),
            total_calculations
        );
        web_sys::console::log_1(&output.into());

        // 開始日時設定（2012-06-15 12:00:00 UTC）
        let base_timestamp = 1339718400i64; // 2012-06-15 12:00:00 UTC
        let base_seconds_since_2000 = base_timestamp - 946684800i64; // EPOCH_2000_UNIX

        let start = Date::now();
        let mut matches_found = 0;
        let mut calculations_done = 0u64;

        // 統合探索のメインループ（WebAssembly部分を除く）
        for second_offset in 0..range_seconds {
            let current_seconds_since_2000 = base_seconds_since_2000 + second_offset as i64;

            if current_seconds_since_2000 < 0 {
                continue;
            }

            // 日時インデックス計算
            let time_index = (current_seconds_since_2000 % 86400) as u32;
            let date_index = (current_seconds_since_2000 / 86400) as u32;

            // 日時コード取得
            let time_code = TimeCodeGenerator::get_time_code_for_hardware(time_index, hardware);
            let date_code = DateCodeGenerator::get_date_code(date_index);

            // Timer0とVCount範囲探索
            for timer0 in 0..timer0_range {
                for vcount in 0..vcount_range {
                    // メッセージ構築
                    let mut message = base_message;
                    message[5] = swap_bytes_32(((vcount as u32) << 16) | (timer0 as u32));
                    message[8] = date_code;
                    message[9] = time_code;

                    // SHA-1計算とSeed生成
                    let (h0, h1, _h2, _h3, _h4) = calculate_pokemon_sha1(&message);
                    let seed = calculate_pokemon_seed_from_hash(h0, h1);

                    // 目標Seedとマッチング
                    for &target in &target_seeds {
                        if seed == target {
                            matches_found += 1;
                        }
                    }

                    calculations_done += 1;
                }
            }
        }

        let end = Date::now();
        let duration_ms = end - start;

        // 結果出力
        web_sys::console::log_1(&"=== 統合Seed探索パフォーマンス結果 ===".into());
        let output = format!("総計算回数: {}", calculations_done);
        web_sys::console::log_1(&output.into());
        let output = format!("実行時間: {:.2}ms", duration_ms);
        web_sys::console::log_1(&output.into());
        let output = format!("発見されたマッチ: {}", matches_found);
        web_sys::console::log_1(&output.into());
        let calc_per_sec = (calculations_done as f64) / (duration_ms / 1000.0);
        let output = format!("1秒あたりの計算数: {:.2} calculations/sec", calc_per_sec);
        web_sys::console::log_1(&output.into());

        if calculations_done > 0 {
            let avg_time_ns = (duration_ms * 1_000_000.0) / calculations_done as f64;
            let output = format!("1回あたりの平均時間: {:.2} ns", avg_time_ns);
            web_sys::console::log_1(&output.into());
        }

        // パフォーマンス基準チェック（実性能の50%程度を基準とする）
        assert!(
            calc_per_sec > 250_000.0,
            "統合探索性能が基準を下回りました: {:.2} calc/sec",
            calc_per_sec
        );

        web_sys::console::log_1(&"=== 統合Seed探索パフォーマンステスト完了 ===".into());
    }

    // SearchResult テスト（元 search_result.rs から移行）
    #[wasm_bindgen_test]
    fn test_search_result_creation_and_getters() {
        let result = SearchResult::new(
            0x12345678,
            "abcdef1234567890abcdef1234567890abcdef12".to_string(),
            2012,
            6,
            15,
            10,
            30,
            45,
            0x2FFF,
            1120,
            50,
        );

        assert_eq!(result.seed(), 0x12345678);
        assert_eq!(result.hash(), "abcdef1234567890abcdef1234567890abcdef12");
        assert_eq!(result.year(), 2012);
        assert_eq!(result.month(), 6);
        assert_eq!(result.date(), 15);
        assert_eq!(result.hour(), 10);
        assert_eq!(result.minute(), 30);
        assert_eq!(result.second(), 45);
        assert_eq!(result.key_code(), 0x2FFF);
        assert_eq!(result.timer0(), 1120);
        assert_eq!(result.vcount(), 50);
    }
}

// ネイティブ環境用テスト
#[cfg(all(test, not(target_arch = "wasm32")))]
mod native_tests {
    use super::*;

    // ==== SearchResult のテスト ====
    // SearchResultはwasm_bindgenに依存していないのでネイティブ環境でもテスト可能

    #[test]
    fn test_search_result() {
        let result = SearchResult::new(
            0x12345678,
            "abcdef1234567890abcdef1234567890abcdef12".to_string(),
            2012,
            6,
            15,
            10,
            30,
            45,
            0x2FFF,
            1120,
            50,
        );
        assert_eq!(result.seed(), 0x12345678);
        assert_eq!(result.hash(), "abcdef1234567890abcdef1234567890abcdef12");
        assert_eq!(result.year(), 2012);
        assert_eq!(result.month(), 6);
        assert_eq!(result.date(), 15);
        assert_eq!(result.hour(), 10);
        assert_eq!(result.minute(), 30);
        assert_eq!(result.second(), 45);
        assert_eq!(result.key_code(), 0x2FFF);
        assert_eq!(result.timer0(), 1120);
        assert_eq!(result.vcount(), 50);
    }

    // 注意: IntegratedSeedSearcherはwasm_bindgen依存のため、
    // ネイティブ環境ではテストできません。
    // WASMテストでのみ実行されます。

    // ==== キーコード生成とフィルタリングのテスト ====

    /// キーコードのビット定義
    const A: u32 = 1 << 0;
    const B: u32 = 1 << 1;
    const SELECT: u32 = 1 << 2;
    const START: u32 = 1 << 3;
    const RIGHT: u32 = 1 << 4;
    const LEFT: u32 = 1 << 5;
    const UP: u32 = 1 << 6;
    const DOWN: u32 = 1 << 7;
    const R: u32 = 1 << 8;
    const L: u32 = 1 << 9;
    const X: u32 = 1 << 10;
    const Y: u32 = 1 << 11;

    /// XOR 0x2FFFする前の生のキーコードをXOR後のキーコードに変換
    fn raw_to_xored(raw: u32) -> u32 {
        raw ^ 0x2FFF
    }

    /// XOR後のキーコードを生のキーコードに戻す（逆変換）
    fn xored_to_raw(xored: u32) -> u32 {
        xored ^ 0x2FFF
    }

    #[test]
    fn test_generate_key_codes_excludes_up_down_combination() {
        // UP + DOWN のビットを有効化
        let mask = UP | DOWN | A; // A も含めて3ビット有効
        let key_codes = crate::integrated_search::generate_key_codes(mask);

        // 生成されたキーコードを元の形式に戻して検証
        for &key_code in &key_codes {
            let raw = xored_to_raw(key_code);
            // UP と DOWN が同時に押されているものは含まれないはず
            assert!(
                !((raw & UP) != 0 && (raw & DOWN) != 0),
                "UP + DOWN 同時押しが検出されました: raw=0x{:X}, key_code=0x{:X}",
                raw,
                key_code
            );
        }

        // 期待される組み合わせ数を確認
        // 3ビット有効 (UP, DOWN, A) で2^3=8通りの組み合わせから、
        // UP+DOWN, UP+DOWN+A の2通りを除外して6通り
        assert_eq!(key_codes.len(), 6);
    }

    #[test]
    fn test_generate_key_codes_excludes_left_right_combination() {
        // LEFT + RIGHT のビットを有効化
        let mask = LEFT | RIGHT | B; // B も含めて3ビット有効
        let key_codes = crate::integrated_search::generate_key_codes(mask);

        // 生成されたキーコードを元の形式に戻して検証
        for &key_code in &key_codes {
            let raw = xored_to_raw(key_code);
            // LEFT と RIGHT が同時に押されているものは含まれないはず
            assert!(
                !((raw & LEFT) != 0 && (raw & RIGHT) != 0),
                "LEFT + RIGHT 同時押しが検出されました: raw=0x{:X}, key_code=0x{:X}",
                raw,
                key_code
            );
        }

        // 期待される組み合わせ数を確認
        assert_eq!(key_codes.len(), 6);
    }

    #[test]
    fn test_generate_key_codes_excludes_start_select_l_r_combination() {
        // START + SELECT + L + R のビットを有効化
        let mask = START | SELECT | L | R;
        let key_codes = crate::integrated_search::generate_key_codes(mask);

        // 生成されたキーコードを元の形式に戻して検証
        for &key_code in &key_codes {
            let raw = xored_to_raw(key_code);
            // START, SELECT, L, R の4つが同時に押されているものは含まれないはず
            let has_all_four =
                (raw & START) != 0 && (raw & SELECT) != 0 && (raw & L) != 0 && (raw & R) != 0;
            assert!(
                !has_all_four,
                "START + SELECT + L + R 4つ同時押しが検出されました: raw=0x{:X}, key_code=0x{:X}",
                raw, key_code
            );
        }

        // 4ビット有効で2^4=16通りの組み合わせから、
        // START+SELECT+L+R の1通りを除外して15通り
        assert_eq!(key_codes.len(), 15);
    }

    #[test]
    fn test_generate_key_codes_complex_combination() {
        // 複数の不可能な組み合わせを含むマスク
        let mask = UP | DOWN | LEFT | RIGHT | START | SELECT | L | R;
        let key_codes = crate::integrated_search::generate_key_codes(mask);

        for &key_code in &key_codes {
            let raw = xored_to_raw(key_code);

            // UP + DOWN 同時押しチェック
            assert!(
                !((raw & UP) != 0 && (raw & DOWN) != 0),
                "UP + DOWN 同時押しが検出されました: raw=0x{:X}",
                raw
            );

            // LEFT + RIGHT 同時押しチェック
            assert!(
                !((raw & LEFT) != 0 && (raw & RIGHT) != 0),
                "LEFT + RIGHT 同時押しが検出されました: raw=0x{:X}",
                raw
            );

            // START + SELECT + L + R 4つ同時押しチェック
            let has_all_four =
                (raw & START) != 0 && (raw & SELECT) != 0 && (raw & L) != 0 && (raw & R) != 0;
            assert!(
                !has_all_four,
                "START + SELECT + L + R 4つ同時押しが検出されました: raw=0x{:X}",
                raw
            );
        }

        // 8ビット有効 (2^8 = 256通り) から不可能な組み合わせを除外
        // 正確な数は計算が複雑だが、256より少なくなることを確認
        assert!(key_codes.len() < 256);
        assert!(key_codes.len() > 0);
    }

    #[test]
    fn test_generate_key_codes_valid_combinations_included() {
        // 有効な組み合わせがちゃんと含まれることを確認
        let mask = UP | DOWN | LEFT | RIGHT;
        let key_codes = crate::integrated_search::generate_key_codes(mask);

        // UP のみ押されている組み合わせが含まれるか
        let up_only = raw_to_xored(UP);
        assert!(
            key_codes.contains(&up_only),
            "UP のみの組み合わせが含まれていません"
        );

        // DOWN のみ押されている組み合わせが含まれるか
        let down_only = raw_to_xored(DOWN);
        assert!(
            key_codes.contains(&down_only),
            "DOWN のみの組み合わせが含まれていません"
        );

        // LEFT のみ押されている組み合わせが含まれるか
        let left_only = raw_to_xored(LEFT);
        assert!(
            key_codes.contains(&left_only),
            "LEFT のみの組み合わせが含まれていません"
        );

        // RIGHT のみ押されている組み合わせが含まれるか
        let right_only = raw_to_xored(RIGHT);
        assert!(
            key_codes.contains(&right_only),
            "RIGHT のみの組み合わせが含まれていません"
        );

        // UP + LEFT の組み合わせが含まれるか（これは有効）
        let up_left = raw_to_xored(UP | LEFT);
        assert!(
            key_codes.contains(&up_left),
            "UP + LEFT の組み合わせが含まれていません"
        );

        // 何も押されていない組み合わせが含まれるか
        let none = raw_to_xored(0);
        assert!(
            key_codes.contains(&none),
            "何も押されていない組み合わせが含まれていません"
        );
    }

    #[test]
    fn test_generate_key_codes_no_mask() {
        // マスクが0の場合（何も有効でない）
        let mask = 0;
        let key_codes = crate::integrated_search::generate_key_codes(mask);

        // 何も押されていない組み合わせのみ
        assert_eq!(key_codes.len(), 1);
        assert_eq!(key_codes[0], 0x2FFF); // 0 XOR 0x2FFF
    }

    #[test]
    fn test_generate_key_codes_single_bit() {
        // 単一ビットのみ有効
        let mask = A;
        let key_codes = crate::integrated_search::generate_key_codes(mask);

        // 2通り（押す、押さない）
        assert_eq!(key_codes.len(), 2);
        assert!(key_codes.contains(&raw_to_xored(0))); // 何も押さない
        assert!(key_codes.contains(&raw_to_xored(A))); // A を押す
    }

    #[test]
    fn test_generate_key_codes_all_invalid_patterns_excluded() {
        // 全ビット有効にして、不可能な組み合わせがすべて除外されることを確認
        let mask = 0xFFF; // 12ビット全て有効
        let key_codes = crate::integrated_search::generate_key_codes(mask);

        // 2^12 = 4096通りから不可能な組み合わせを除外
        let total_combinations = 1 << 12;
        assert!(key_codes.len() < total_combinations);

        // 全ての生成されたキーコードが有効であることを確認
        for &key_code in &key_codes {
            let raw = xored_to_raw(key_code);

            // UP + DOWN 同時押しがないことを確認
            assert!(
                !((raw & UP) != 0 && (raw & DOWN) != 0),
                "UP + DOWN 同時押しが検出: 0x{:X}",
                raw
            );

            // LEFT + RIGHT 同時押しがないことを確認
            assert!(
                !((raw & LEFT) != 0 && (raw & RIGHT) != 0),
                "LEFT + RIGHT 同時押しが検出: 0x{:X}",
                raw
            );

            // START + SELECT + L + R の4つ同時押しがないことを確認
            let has_all_four =
                (raw & START) != 0 && (raw & SELECT) != 0 && (raw & L) != 0 && (raw & R) != 0;
            assert!(
                !has_all_four,
                "START + SELECT + L + R 4つ同時押しが検出: 0x{:X}",
                raw
            );
        }

        // 除外された組み合わせの数を確認
        let excluded_count = total_combinations - key_codes.len();
        // 最低でも何らかの組み合わせが除外されているはず
        assert!(excluded_count > 0, "不可能な組み合わせが除外されていません");
    }

    #[test]
    fn test_specific_invalid_combinations() {
        // 特定の不可能な組み合わせが生成されないことを直接テスト

        // UP + DOWN のみを有効化
        let mask_up_down = UP | DOWN;
        let key_codes = crate::integrated_search::generate_key_codes(mask_up_down);
        // UP+DOWN の組み合わせが含まれないことを確認
        let up_down_combination = raw_to_xored(UP | DOWN);
        assert!(
            !key_codes.contains(&up_down_combination),
            "UP + DOWN の組み合わせが含まれています"
        );

        // LEFT + RIGHT のみを有効化
        let mask_left_right = LEFT | RIGHT;
        let key_codes = crate::integrated_search::generate_key_codes(mask_left_right);
        // LEFT+RIGHT の組み合わせが含まれないことを確認
        let left_right_combination = raw_to_xored(LEFT | RIGHT);
        assert!(
            !key_codes.contains(&left_right_combination),
            "LEFT + RIGHT の組み合わせが含まれています"
        );

        // START + SELECT + L + R のみを有効化
        let mask_four_buttons = START | SELECT | L | R;
        let key_codes = crate::integrated_search::generate_key_codes(mask_four_buttons);
        // 4つの組み合わせが含まれないことを確認
        let four_buttons_combination = raw_to_xored(START | SELECT | L | R);
        assert!(
            !key_codes.contains(&four_buttons_combination),
            "START + SELECT + L + R の組み合わせが含まれています"
        );
    }
}

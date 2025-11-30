//! MtSeedBootTimingSearchIterator パフォーマンステスト
//!
//! 起動時間検索のスループットを測定する。

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_browser);

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests {
    use super::*;
    use crate::mt_seed_boot_timing_search::MtSeedBootTimingSearchIterator;
    use crate::search_common::{DSConfigJs, SearchRangeParamsJs, SegmentParamsJs, TimeRangeParamsJs};
    use js_sys::Date;
    use std::collections::HashSet;

    /// テスト用のパラメータ定義
    const TEST_MAC: [u8; 6] = [0x00, 0x09, 0xBF, 0xAA, 0xBB, 0xCC];
    const TEST_NAZO: [u32; 5] = [0x02215F10, 0x02215F30, 0x02215F20, 0x02761008, 0x00000000];
    const TEST_TIMER0: u32 = 0x1000;
    const TEST_VCOUNT: u32 = 0x60;
    const TEST_KEY_CODE: u32 = 0x2FFF;

    /// ログ出力ヘルパー
    fn log(msg: &str) {
        web_sys::console::log_1(&msg.into());
    }

    /// パフォーマンス結果を表示
    fn log_performance(label: &str, iterations: u64, duration_ms: f64) {
        let rate = iterations as f64 / (duration_ms / 1000.0);
        let ns_per_iter = (duration_ms * 1_000_000.0) / iterations as f64;
        log(&format!(
            "[{label}] {iterations} iterations in {duration_ms:.2}ms = {rate:.0} iter/sec ({ns_per_iter:.1} ns/iter)"
        ));
    }

    // =========================================================================
    // MtSeedBootTimingSearchIterator のパフォーマンス測定
    // =========================================================================

    #[wasm_bindgen_test]
    fn test_mt_seed_boot_timing_iterator_performance() {
        log("=== MtSeedBootTimingSearchIterator パフォーマンステスト ===");

        let range_seconds = 86400u32; // 1日分
        let target_seeds: Vec<u32> = vec![0x12345678, 0xABCDEF00];

        // パラメータ構築
        let ds_config =
            DSConfigJs::new(&TEST_MAC, &TEST_NAZO, "DS").expect("Failed to create DSConfigJs");
        let segment = SegmentParamsJs::new(TEST_TIMER0, TEST_VCOUNT, TEST_KEY_CODE);
        let time_range =
            TimeRangeParamsJs::new(0, 23, 0, 59, 0, 59).expect("Failed to create TimeRangeParamsJs");
        let search_range = SearchRangeParamsJs::new(2024, 1, 1, range_seconds)
            .expect("Failed to create SearchRangeParamsJs");

        // ウォームアップ
        {
            let warmup_range = SearchRangeParamsJs::new(2024, 1, 1, 1000).unwrap();
            let mut warmup_iter = MtSeedBootTimingSearchIterator::new(
                &ds_config,
                &segment,
                &time_range,
                &warmup_range,
                &target_seeds,
            )
            .expect("Failed to create warmup iterator");

            while !warmup_iter.is_finished() {
                let _ = warmup_iter.next_batch(100, 1000);
            }
        }

        // 計測
        let start = Date::now();

        let mut iterator = MtSeedBootTimingSearchIterator::new(
            &ds_config,
            &segment,
            &time_range,
            &search_range,
            &target_seeds,
        )
        .expect("Failed to create MtSeedBootTimingSearchIterator");

        let mut total_processed = 0u32;
        while !iterator.is_finished() {
            let results = iterator.next_batch(1000, 10000); // 10000秒ずつ処理
            total_processed = iterator.processed_seconds();
            let _ = results.length(); // 結果を消費
        }

        let duration = Date::now() - start;

        log_performance("MtSeedBootTimingIterator (1 segment)", total_processed as u64, duration);

        // 複数セグメントテスト（セグメントループをシミュレート）
        let timer0_count = 3u32;
        let vcount_count = 2u32;
        let segment_count = timer0_count * vcount_count;

        let start = Date::now();
        let mut total_iterations = 0u64;

        for t0_offset in 0..timer0_count {
            for vc_offset in 0..vcount_count {
                let segment = SegmentParamsJs::new(
                    TEST_TIMER0 + t0_offset,
                    TEST_VCOUNT + vc_offset,
                    TEST_KEY_CODE,
                );

                let mut iterator = MtSeedBootTimingSearchIterator::new(
                    &ds_config,
                    &segment,
                    &time_range,
                    &search_range,
                    &target_seeds,
                )
                .expect("Failed to create iterator");

                while !iterator.is_finished() {
                    let _ = iterator.next_batch(1000, 10000);
                }

                total_iterations += iterator.processed_seconds() as u64;
            }
        }

        let duration = Date::now() - start;

        log_performance(
            &format!("MtSeedBootTimingIterator ({segment_count} segments)"),
            total_iterations,
            duration,
        );

        log("=== MtSeedBootTimingSearchIterator パフォーマンステスト完了 ===");
    }

    // =========================================================================
    // コンポーネント別パフォーマンス分析
    // =========================================================================

    #[wasm_bindgen_test]
    fn test_component_breakdown() {
        log("=== コンポーネント別パフォーマンス分析 ===");

        let iterations = 100_000u32;

        // 1. HashValuesEnumerator のイテレーション速度
        {
            let ds_config = DSConfigJs::new(&TEST_MAC, &TEST_NAZO, "DS").unwrap();
            let segment = SegmentParamsJs::new(TEST_TIMER0, TEST_VCOUNT, TEST_KEY_CODE);
            let time_range = TimeRangeParamsJs::new(0, 23, 0, 59, 0, 59).unwrap();
            let search_range = SearchRangeParamsJs::new(2024, 1, 1, iterations).unwrap();

            // NOTE: MtSeedBootTimingSearchIteratorは空のtarget_seedsを拒否するため、
            // ダミーseedを使う
            let dummy_seeds: Vec<u32> = vec![0xFFFFFFFF];

            let start = Date::now();

            let mut iterator = MtSeedBootTimingSearchIterator::new(
                &ds_config,
                &segment,
                &time_range,
                &search_range,
                &dummy_seeds,
            )
            .expect("Failed to create iterator");

            while !iterator.is_finished() {
                let _ = iterator.next_batch(1000, iterations);
            }

            let duration = Date::now() - start;
            log_performance("HashValuesEnumerator iteration", iterations as u64, duration);
        }

        // 2. SHA-1計算（直接比較用）
        {
            use crate::sha1::calculate_pokemon_sha1;
            use crate::sha1_simd::calculate_pokemon_sha1_simd;

            let test_message: [u32; 16] = [
                0x02215F10, 0x02215F30, 0x02215F20, 0x02761008, 0x00000000, 0x12345678, 0x9ABCDEF0,
                0x34561234, 0x0C0F0F04, 0x00120000, 0x00000000, 0x00000000, 0x00000005, 0x80000000,
                0x00000000, 0x000001A0,
            ];

            // スカラー
            let scalar_iterations = iterations;
            let start = Date::now();
            let mut checksum = 0u64;
            for i in 0..scalar_iterations {
                let mut msg = test_message;
                msg[8] = msg[8].wrapping_add(i);
                let hash = calculate_pokemon_sha1(&msg);
                checksum = checksum.wrapping_add(hash.h0 as u64).wrapping_add(hash.h1 as u64);
            }
            let duration = Date::now() - start;
            log_performance("SHA-1 scalar", scalar_iterations as u64, duration);
            log(&format!("  checksum: 0x{checksum:016X}"));

            // SIMD (4並列)
            let simd_batches = iterations / 4;
            let start = Date::now();
            let mut checksum = 0u64;
            for i in 0..simd_batches {
                let mut messages = [0u32; 64];
                for lane in 0..4 {
                    let mut msg = test_message;
                    msg[8] = msg[8].wrapping_add(i * 4 + lane);
                    messages[lane as usize * 16..(lane as usize + 1) * 16].copy_from_slice(&msg);
                }
                let results = calculate_pokemon_sha1_simd(&messages);
                for lane in 0..4 {
                    checksum = checksum
                        .wrapping_add(results[lane].h0 as u64)
                        .wrapping_add(results[lane].h1 as u64);
                }
            }
            let duration = Date::now() - start;
            log_performance("SHA-1 SIMD (4-way)", (simd_batches * 4) as u64, duration);
            log(&format!("  checksum: 0x{checksum:016X}"));
        }

        log("=== コンポーネント別パフォーマンス分析完了 ===");
    }

    // =========================================================================
    // HashSet vs BTreeSet ルックアップ比較
    // =========================================================================

    #[wasm_bindgen_test]
    fn test_seed_lookup_performance() {
        log("=== Seed ルックアップパフォーマンス比較 ===");

        use std::collections::BTreeSet;

        let iterations = 1_000_000u32;
        let target_count = 100;

        // テストデータ生成
        let targets: Vec<u32> = (0..target_count).map(|i| i * 12345 + 67890).collect();
        let hash_set: HashSet<u32> = targets.iter().copied().collect();
        let btree_set: BTreeSet<u32> = targets.iter().copied().collect();

        // HashSet
        let start = Date::now();
        let mut hits = 0u64;
        for i in 0..iterations {
            let test_seed = i.wrapping_mul(7919);
            if hash_set.contains(&test_seed) {
                hits += 1;
            }
        }
        let duration = Date::now() - start;
        log_performance("HashSet lookup", iterations as u64, duration);
        log(&format!("  hits: {hits}"));

        // BTreeSet
        let start = Date::now();
        let mut hits = 0u64;
        for i in 0..iterations {
            let test_seed = i.wrapping_mul(7919);
            if btree_set.contains(&test_seed) {
                hits += 1;
            }
        }
        let duration = Date::now() - start;
        log_performance("BTreeSet lookup", iterations as u64, duration);
        log(&format!("  hits: {hits}"));

        log("=== Seed ルックアップパフォーマンス比較完了 ===");
    }
}

// ネイティブ環境用テスト（基本的な型チェックのみ）
#[cfg(all(test, not(target_arch = "wasm32")))]
mod native_tests {
    use std::collections::{BTreeSet, HashSet};

    #[test]
    fn test_hashset_vs_btreeset_native() {
        let iterations = 100_000u32;
        let target_count = 100;

        let targets: Vec<u32> = (0..target_count).map(|i| i * 12345 + 67890).collect();
        let hash_set: HashSet<u32> = targets.iter().copied().collect();
        let btree_set: BTreeSet<u32> = targets.iter().copied().collect();

        let start = std::time::Instant::now();
        let mut hash_hits = 0u64;
        for i in 0..iterations {
            let test_seed = i.wrapping_mul(7919);
            if hash_set.contains(&test_seed) {
                hash_hits += 1;
            }
        }
        let hash_duration = start.elapsed();

        let start = std::time::Instant::now();
        let mut btree_hits = 0u64;
        for i in 0..iterations {
            let test_seed = i.wrapping_mul(7919);
            if btree_set.contains(&test_seed) {
                btree_hits += 1;
            }
        }
        let btree_duration = start.elapsed();

        println!("HashSet:  {:?} ({} hits)", hash_duration, hash_hits);
        println!("BTreeSet: {:?} ({} hits)", btree_duration, btree_hits);

        assert_eq!(hash_hits, btree_hits);
    }

    #[test]
    fn test_hash_values_enumerator_performance() {
        use crate::search_common::{
            build_ranged_time_code_table, BaseMessageBuilder, HashValuesEnumerator,
            DSConfig, HardwareType, SegmentParams, TimeRangeParams,
        };

        let mac = [0x00, 0x09, 0xBF, 0xAA, 0xBB, 0xCC];
        let nazo = [0x02215F10, 0x02215F30, 0x02215F20, 0x02761008, 0x00000000];
        let ds_config = DSConfig::new(mac, nazo, HardwareType::DS);
        let segment = SegmentParams::new(0x1000, 0x60, 0x2FFF);
        
        let time_range = TimeRangeParams::new(0, 23, 0, 59, 0, 59).unwrap();
        let iterations = 86400u32; // 1日分

        // next_quad APIテスト
        {
            let builder = BaseMessageBuilder::from_params(&ds_config, &segment);
            let table = build_ranged_time_code_table(&time_range, HardwareType::DS);
            let mut enumerator = HashValuesEnumerator::new(builder, table, 0, iterations);

            let start = std::time::Instant::now();
            let mut count = 0u64;
            let mut checksum = 0u64;
            loop {
                let (entries, len) = enumerator.next_quad();
                if len == 0 {
                    break;
                }
                for i in 0..len as usize {
                    count += 1;
                    checksum = checksum.wrapping_add(entries[i].hash.h0 as u64);
                }
            }
            let duration = start.elapsed();
            let rate = count as f64 / duration.as_secs_f64();
            
            println!("Iterator API: {:?} ({} entries, {:.0} iter/sec)", duration, count, rate);
            println!("  checksum: 0x{:016X}", checksum);
        }
    }

    #[test]
    fn test_hash_enumerator_performance_comparison() {
        use crate::search_common::{
            build_ranged_time_code_table, BaseMessageBuilder, HashValuesEnumerator,
            DSConfig, HardwareType, SegmentParams, TimeRangeParams,
        };

        let mac = [0x00, 0x09, 0xBF, 0xAA, 0xBB, 0xCC];
        let nazo = [0x02215F10, 0x02215F30, 0x02215F20, 0x02761008, 0x00000000];
        let target_seeds: Vec<u32> = vec![0x12345678, 0xABCDEF00];
        let range_seconds = 86400u32; // 1日分

        println!("\n=== HashValuesEnumerator パフォーマンス比較 ===");

        // 1. next_quad API (シード検索込み)
        {
            let ds_config = DSConfig::new(mac, nazo, HardwareType::DS);
            let segment = SegmentParams::new(0x1000, 0x60, 0x2FFF);
            let time_range = TimeRangeParams::new(0, 23, 0, 59, 0, 59).unwrap();
            let target_set: HashSet<u32> = target_seeds.iter().copied().collect();

            let builder = BaseMessageBuilder::from_params(&ds_config, &segment);
            let table = build_ranged_time_code_table(&time_range, HardwareType::DS);
            let mut enumerator = HashValuesEnumerator::new(builder, table, 0, range_seconds);

            let start = std::time::Instant::now();
            let mut found = 0u32;
            loop {
                let (entries, len) = enumerator.next_quad();
                if len == 0 {
                    break;
                }
                for i in 0..len as usize {
                    let mt_seed = entries[i].hash.to_mt_seed();
                    if target_set.contains(&mt_seed) {
                        found += 1;
                    }
                }
            }
            let duration = start.elapsed();
            let rate = range_seconds as f64 / duration.as_secs_f64();
            
            println!("Iterator API (with lookup): {:?} ({:.0} iter/sec, {} found)", duration, rate, found);
        }

        println!("=== 比較完了 ===\n");
    }
}

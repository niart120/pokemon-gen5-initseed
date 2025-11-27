//! IVBootTimingSearchIterator と IntegratedSeedSearcher のパフォーマンス比較テスト
//!
//! 両者のスループットを測定し、リファクタリングによる性能差を定量化する。

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_browser);

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests {
    use super::*;
    use crate::integrated_search::IntegratedSeedSearcher;
    use crate::iv_boot_timing_search::IVBootTimingSearchIterator;
    use crate::search_common::{DSConfigJs, SearchRangeParamsJs, SegmentParamsJs, TimeRangeParamsJs};
    use js_sys::Date;
    use std::collections::HashSet;

    /// テスト用のパラメータ定義
    const TEST_MAC: [u8; 6] = [0x00, 0x09, 0xBF, 0xAA, 0xBB, 0xCC];
    const TEST_NAZO: [u32; 5] = [0x02215F10, 0x02215F30, 0x02215F20, 0x02761008, 0x00000000];
    const TEST_TIMER0: u32 = 0x1000;
    const TEST_VCOUNT: u32 = 0x60;
    const TEST_KEY_CODE: u32 = 0x2FFF;
    const TEST_FRAME: u32 = 8; // DS

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
    // IntegratedSeedSearcher のベースラインパフォーマンス測定
    // =========================================================================

    #[wasm_bindgen_test]
    fn test_baseline_integrated_searcher_performance() {
        log("=== IntegratedSeedSearcher ベースラインテスト ===");

        // 1日分の検索（86400秒）、単一Timer0/VCount/KeyCode
        let range_seconds = 86400u32;
        let target_seeds: Vec<u32> = vec![0x12345678, 0xABCDEF00]; // ダミー

        let searcher = IntegratedSeedSearcher::new(
            &TEST_MAC,
            &TEST_NAZO,
            "DS",
            0, // key_input_mask = 0 → keyCode = 0x2FFF のみ
            TEST_FRAME,
            0, 23, // hour
            0, 59, // minute
            0, 59, // second
        )
        .expect("Failed to create IntegratedSeedSearcher");

        // ウォームアップ
        let _ = searcher.search_seeds_integrated_simd(
            2024, 1, 1, 0, 0, 0, 1000, TEST_TIMER0, TEST_TIMER0, TEST_VCOUNT, TEST_VCOUNT, &target_seeds,
        );

        // 計測
        let start = Date::now();

        let _results = searcher.search_seeds_integrated_simd(
            2024,
            1,
            1,
            0,
            0,
            0,
            range_seconds,
            TEST_TIMER0,
            TEST_TIMER0, // 単一Timer0
            TEST_VCOUNT,
            TEST_VCOUNT, // 単一VCount
            &target_seeds,
        );

        let duration = Date::now() - start;

        log_performance("IntegratedSearcher (1 segment)", range_seconds as u64, duration);

        // 複数セグメントテスト（Timer0 × 3, VCount × 2 = 6セグメント）
        let timer0_count = 3u32;
        let vcount_count = 2u32;
        let total_iterations = range_seconds as u64 * timer0_count as u64 * vcount_count as u64;

        let start = Date::now();

        let _results = searcher.search_seeds_integrated_simd(
            2024,
            1,
            1,
            0,
            0,
            0,
            range_seconds,
            TEST_TIMER0,
            TEST_TIMER0 + timer0_count - 1,
            TEST_VCOUNT,
            TEST_VCOUNT + vcount_count - 1,
            &target_seeds,
        );

        let duration = Date::now() - start;

        log_performance(
            &format!("IntegratedSearcher ({} segments)", timer0_count * vcount_count),
            total_iterations,
            duration,
        );

        log("=== IntegratedSeedSearcher ベースラインテスト完了 ===");
    }

    // =========================================================================
    // IVBootTimingSearchIterator のパフォーマンス測定
    // =========================================================================

    #[wasm_bindgen_test]
    fn test_iv_boot_timing_iterator_performance() {
        log("=== IVBootTimingSearchIterator パフォーマンステスト ===");

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
            let mut warmup_iter = IVBootTimingSearchIterator::new(
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

        let mut iterator = IVBootTimingSearchIterator::new(
            &ds_config,
            &segment,
            &time_range,
            &search_range,
            &target_seeds,
        )
        .expect("Failed to create IVBootTimingSearchIterator");

        let mut total_processed = 0u32;
        while !iterator.is_finished() {
            let results = iterator.next_batch(1000, 10000); // 10000秒ずつ処理
            total_processed = iterator.processed_seconds();
            let _ = results.length(); // 結果を消費
        }

        let duration = Date::now() - start;

        log_performance("IVBootTimingIterator (1 segment)", total_processed as u64, duration);

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

                let mut iterator = IVBootTimingSearchIterator::new(
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
            &format!("IVBootTimingIterator ({segment_count} segments)"),
            total_iterations,
            duration,
        );

        log("=== IVBootTimingSearchIterator パフォーマンステスト完了 ===");
    }

    // =========================================================================
    // 直接比較テスト（同一条件で両者を測定）
    // =========================================================================

    #[wasm_bindgen_test]
    fn test_performance_comparison() {
        log("=== パフォーマンス直接比較テスト ===");

        // 共通パラメータ
        let range_seconds = 86400u32; // 1日
        let timer0_count = 3u32;
        let vcount_count = 2u32;
        let target_seeds: Vec<u32> = vec![0x12345678, 0xABCDEF00, 0x11111111, 0x22222222];

        let total_iterations =
            range_seconds as u64 * timer0_count as u64 * vcount_count as u64;

        log(&format!(
            "条件: {} 秒 × Timer0({}) × VCount({}) = {} iterations",
            range_seconds, timer0_count, vcount_count, total_iterations
        ));

        // --- IntegratedSeedSearcher ---
        let integrated_duration = {
            let searcher = IntegratedSeedSearcher::new(
                &TEST_MAC,
                &TEST_NAZO,
                "DS",
                0,
                TEST_FRAME,
                0,
                23,
                0,
                59,
                0,
                59,
            )
            .expect("Failed to create IntegratedSeedSearcher");

            let start = Date::now();

            let _results = searcher.search_seeds_integrated_simd(
                2024,
                1,
                1,
                0,
                0,
                0,
                range_seconds,
                TEST_TIMER0,
                TEST_TIMER0 + timer0_count - 1,
                TEST_VCOUNT,
                TEST_VCOUNT + vcount_count - 1,
                &target_seeds,
            );

            Date::now() - start
        };

        // --- IVBootTimingSearchIterator ---
        let iterator_duration = {
            let ds_config = DSConfigJs::new(&TEST_MAC, &TEST_NAZO, "DS").unwrap();
            let time_range = TimeRangeParamsJs::new(0, 23, 0, 59, 0, 59).unwrap();
            let search_range = SearchRangeParamsJs::new(2024, 1, 1, range_seconds).unwrap();

            let start = Date::now();

            for t0_offset in 0..timer0_count {
                for vc_offset in 0..vcount_count {
                    let segment = SegmentParamsJs::new(
                        TEST_TIMER0 + t0_offset,
                        TEST_VCOUNT + vc_offset,
                        TEST_KEY_CODE,
                    );

                    let mut iterator = IVBootTimingSearchIterator::new(
                        &ds_config,
                        &segment,
                        &time_range,
                        &search_range,
                        &target_seeds,
                    )
                    .expect("Failed to create iterator");

                    while !iterator.is_finished() {
                        let _ = iterator.next_batch(1000, 86400);
                    }
                }
            }

            Date::now() - start
        };

        // 結果出力
        log("=== 比較結果 ===");
        log_performance("IntegratedSearcher", total_iterations, integrated_duration);
        log_performance("IVBootTimingIterator", total_iterations, iterator_duration);

        let ratio = iterator_duration / integrated_duration;
        log(&format!("性能比: IVBootTimingIterator / IntegratedSearcher = {ratio:.2}x"));

        if ratio > 1.0 {
            log(&format!(
                "⚠ IVBootTimingIterator は IntegratedSearcher より {:.1}% 遅い",
                (ratio - 1.0) * 100.0
            ));
        } else {
            log(&format!(
                "✓ IVBootTimingIterator は IntegratedSearcher より {:.1}% 速い",
                (1.0 - ratio) * 100.0
            ));
        }

        log("=== パフォーマンス直接比較テスト完了 ===");
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
            let target_seeds: Vec<u32> = vec![]; // 空 → マッチなし

            // NOTE: IVBootTimingSearchIteratorは空のtarget_seedsを拒否するため、
            // ダミーseedを使う
            let dummy_seeds: Vec<u32> = vec![0xFFFFFFFF];

            let start = Date::now();

            let mut iterator = IVBootTimingSearchIterator::new(
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
                let (h0, h1, _, _, _) = calculate_pokemon_sha1(&msg);
                checksum = checksum.wrapping_add(h0 as u64).wrapping_add(h1 as u64);
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
                        .wrapping_add(results[lane * 5] as u64)
                        .wrapping_add(results[lane * 5 + 1] as u64);
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
}

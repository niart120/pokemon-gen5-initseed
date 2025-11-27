//! IV起動時間検索器
//!
//! BW/BW2におけるIV確定のための起動時間検索機能を提供する。
//! 複数のtarget_seedsに対してマッチする起動条件を列挙する。
//!
//! ## セグメントパターン設計
//!
//! TypeScript側で timer0 × vcount × keyCode のセグメントループを実装:
//! 1. `IVBootTimingSearchIterator::new()` で単一セグメント（固定timer0/vcount/keyCode）のイテレータを作成
//! 2. `IVBootTimingSearchIterator::next_batch()` で seconds 方向の結果をバッチ取得
//! 3. `is_finished` で完了判定
//!
//! ## 公開API
//! - `IVBootTimingSearchIterator`: 単一セグメントの検索イテレータ
//! - `IVBootTimingSearchResult`: 検索結果1件
//! - `IVBootTimingSearchResults`: バッチ結果

use crate::search_common::{
    build_ranged_time_code_table, generate_display_datetime, BaseMessageBuilder, DSConfigJs,
    HardwareType, RangedTimeCodeTable, SearchRangeParamsJs, SegmentParamsJs, TimeRangeParamsJs,
    SECONDS_PER_DAY,
};
use crate::sha1::calculate_pokemon_seed_from_hash;
use crate::sha1_simd::calculate_pokemon_sha1_simd;
use std::collections::HashSet;
use wasm_bindgen::prelude::*;

// =============================================================================
// 検索結果
// =============================================================================

/// IV起動時間検索結果1件
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct IVBootTimingSearchResult {
    // MT Seed (IV用)
    mt_seed: u32,

    // LCG Seed
    lcg_seed_high: u32,
    lcg_seed_low: u32,

    // 起動条件
    year: u32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
    timer0: u32,
    vcount: u32,
    key_code: u32,
}

#[wasm_bindgen]
impl IVBootTimingSearchResult {
    // MT Seed (IV用)
    #[wasm_bindgen(getter = mtSeed)]
    pub fn mt_seed(&self) -> u32 {
        self.mt_seed
    }

    #[wasm_bindgen(getter = mtSeedHex)]
    pub fn mt_seed_hex(&self) -> String {
        format!("{:08X}", self.mt_seed)
    }

    // LCG Seed
    #[wasm_bindgen(getter = lcgSeedHigh)]
    pub fn lcg_seed_high(&self) -> u32 {
        self.lcg_seed_high
    }

    #[wasm_bindgen(getter = lcgSeedLow)]
    pub fn lcg_seed_low(&self) -> u32 {
        self.lcg_seed_low
    }

    #[wasm_bindgen(getter = lcgSeedHex)]
    pub fn lcg_seed_hex(&self) -> String {
        let lcg_seed = ((self.lcg_seed_high as u64) << 32) | (self.lcg_seed_low as u64);
        format!("{:016X}", lcg_seed)
    }

    // 起動日時
    #[wasm_bindgen(getter)]
    pub fn year(&self) -> u32 {
        self.year
    }

    #[wasm_bindgen(getter)]
    pub fn month(&self) -> u32 {
        self.month
    }

    #[wasm_bindgen(getter)]
    pub fn day(&self) -> u32 {
        self.day
    }

    #[wasm_bindgen(getter)]
    pub fn hour(&self) -> u32 {
        self.hour
    }

    #[wasm_bindgen(getter)]
    pub fn minute(&self) -> u32 {
        self.minute
    }

    #[wasm_bindgen(getter)]
    pub fn second(&self) -> u32 {
        self.second
    }

    // 起動パラメータ
    #[wasm_bindgen(getter)]
    pub fn timer0(&self) -> u32 {
        self.timer0
    }

    #[wasm_bindgen(getter)]
    pub fn vcount(&self) -> u32 {
        self.vcount
    }

    #[wasm_bindgen(getter = keyCode)]
    pub fn key_code(&self) -> u32 {
        self.key_code
    }
}

/// バッチ検索結果
#[wasm_bindgen]
pub struct IVBootTimingSearchResults {
    results: Vec<IVBootTimingSearchResult>,
    processed_in_chunk: u32,
}

#[wasm_bindgen]
impl IVBootTimingSearchResults {
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.results.len()
    }

    #[wasm_bindgen(getter = processedInChunk)]
    pub fn processed_in_chunk(&self) -> u32 {
        self.processed_in_chunk
    }

    /// 結果をJavaScript配列として取得
    #[wasm_bindgen]
    pub fn to_array(&self) -> js_sys::Array {
        let arr = js_sys::Array::new();
        for result in &self.results {
            arr.push(&JsValue::from(result.clone()));
        }
        arr
    }

    /// 指定インデックスの結果を取得
    #[wasm_bindgen]
    pub fn get(&self, index: usize) -> Option<IVBootTimingSearchResult> {
        self.results.get(index).cloned()
    }
}

// =============================================================================
// 検索イテレータ
// =============================================================================

/// IV起動時間検索イテレータ
///
/// 単一セグメント（固定 timer0/vcount/keyCode）に対して seconds 方向の検索を行う。
/// TypeScript側で timer0 × vcount × keyCode のセグメントループを実装し、
/// 各セグメントに対してこのイテレータを作成する。
#[wasm_bindgen]
pub struct IVBootTimingSearchIterator {
    // 内部状態
    base_message_builder: BaseMessageBuilder,
    time_code_table: RangedTimeCodeTable,
    hardware: HardwareType,

    // セグメントパラメータ（結果出力用に保持）
    timer0: u32,
    vcount: u32,
    key_code: u32,

    // 検索条件（複数Seed対応）
    target_seeds: HashSet<u32>,

    // 検索範囲
    start_seconds: i64,
    range_seconds: u32,

    // 現在位置
    current_offset: u32,
    finished: bool,
}

#[wasm_bindgen]
impl IVBootTimingSearchIterator {
    /// コンストラクタ
    ///
    /// # Arguments
    /// - `ds_config`: DS設定パラメータ (MAC/Nazo/Hardware)
    /// - `segment`: セグメントパラメータ (Timer0/VCount/KeyCode)
    /// - `time_range`: 時刻範囲パラメータ
    /// - `search_range`: 検索範囲パラメータ
    /// - `target_seeds`: 検索対象のSeed値（複数可）
    #[wasm_bindgen(constructor)]
    pub fn new(
        ds_config: &DSConfigJs,
        segment: &SegmentParamsJs,
        time_range: &TimeRangeParamsJs,
        search_range: &SearchRangeParamsJs,
        target_seeds: &[u32],
    ) -> Result<IVBootTimingSearchIterator, String> {
        if target_seeds.is_empty() {
            return Err("target_seeds must not be empty".to_string());
        }

        // 内部型に変換
        let ds_config_internal = ds_config.to_ds_config();
        let segment_internal = segment.to_segment_params();
        let time_range_internal = time_range.to_time_range_params();
        let hardware = ds_config_internal.hardware;

        // BaseMessageBuilder構築
        let base_message_builder =
            BaseMessageBuilder::from_params(&ds_config_internal, &segment_internal);

        // RangedTimeCodeTable構築
        let time_code_table = build_ranged_time_code_table(&time_range_internal, hardware);

        // 開始秒を計算
        let start_seconds = search_range.start_seconds_since_2000();

        // target_seedsをHashSetに変換（高速な検索のため）
        let target_seeds_set: HashSet<u32> = target_seeds.iter().copied().collect();

        Ok(IVBootTimingSearchIterator {
            base_message_builder,
            time_code_table,
            hardware,
            timer0: segment_internal.timer0,
            vcount: segment_internal.vcount,
            key_code: segment_internal.key_code,
            target_seeds: target_seeds_set,
            start_seconds,
            range_seconds: search_range.range_seconds(),
            current_offset: 0,
            finished: false,
        })
    }

    /// 検索が完了したかどうか
    #[wasm_bindgen(getter = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// 処理済み秒数
    #[wasm_bindgen(getter = processedSeconds)]
    pub fn processed_seconds(&self) -> u32 {
        self.current_offset
    }

    /// 総秒数
    #[wasm_bindgen(getter = totalSeconds)]
    pub fn total_seconds(&self) -> u32 {
        self.range_seconds
    }

    /// 進捗率 (0.0 - 1.0)
    #[wasm_bindgen(getter)]
    pub fn progress(&self) -> f64 {
        if self.range_seconds == 0 {
            return 1.0;
        }
        self.current_offset as f64 / self.range_seconds as f64
    }

    /// 次のバッチを取得
    ///
    /// - max_results件見つかったら即return
    /// - chunk_seconds秒分処理したら結果がなくても一旦return
    /// - 検索範囲を全て処理したらfinished=trueになる
    #[wasm_bindgen]
    pub fn next_batch(&mut self, max_results: u32, chunk_seconds: u32) -> IVBootTimingSearchResults {
        if self.finished {
            return IVBootTimingSearchResults {
                results: Vec::new(),
                processed_in_chunk: 0,
            };
        }

        let mut results: Vec<IVBootTimingSearchResult> = Vec::new();
        let mut seconds_processed: u32 = 0;

        // SIMD バッチ処理用バッファ
        let mut messages = [0u32; 64]; // 4メッセージ × 16ワード
        let mut batch_metadata: [(i64, u32, u32); 4] = [(0, 0, 0); 4]; // (seconds_since_2000, date_code, time_code)
        let mut batch_len = 0usize;

        let hardware_str = self.hardware.as_str();

        while self.current_offset < self.range_seconds {
            // 終了条件チェック
            if results.len() >= max_results as usize {
                break;
            }
            if seconds_processed >= chunk_seconds {
                break;
            }

            let seconds_since_2000 = self.start_seconds + self.current_offset as i64;
            self.current_offset += 1;
            seconds_processed += 1;

            // 日内秒数を計算
            let second_of_day = (seconds_since_2000.rem_euclid(SECONDS_PER_DAY)) as usize;

            // 許可された秒かチェック + time_code取得
            let time_code = match self.time_code_table[second_of_day] {
                Some(tc) => tc,
                None => continue,
            };

            // date_code計算
            let date_index = (seconds_since_2000 / SECONDS_PER_DAY) as u32;
            let date_code = crate::datetime_codes::DateCodeGenerator::get_date_code(date_index);

            // メッセージをバッファに追加
            let message = self.base_message_builder.build_message(date_code, time_code);
            let offset = batch_len * 16;
            messages[offset..offset + 16].copy_from_slice(&message);
            batch_metadata[batch_len] = (seconds_since_2000, date_code, time_code);
            batch_len += 1;

            // 4つ溜まったらSIMD処理
            if batch_len == 4 {
                self.process_simd_batch(
                    &messages,
                    &batch_metadata,
                    hardware_str,
                    &mut results,
                );
                batch_len = 0;
            }
        }

        // 残りをスカラー処理
        for i in 0..batch_len {
            let offset = i * 16;
            let mut message = [0u32; 16];
            message.copy_from_slice(&messages[offset..offset + 16]);
            let (seconds_since_2000, _, _) = batch_metadata[i];

            self.process_scalar(
                &message,
                seconds_since_2000,
                hardware_str,
                &mut results,
            );
        }

        // 検索完了チェック
        if self.current_offset >= self.range_seconds {
            self.finished = true;
        }

        IVBootTimingSearchResults {
            results,
            processed_in_chunk: seconds_processed,
        }
    }

    /// SIMDバッチ処理
    fn process_simd_batch(
        &self,
        messages: &[u32; 64],
        batch_metadata: &[(i64, u32, u32); 4],
        _hardware_str: &str,
        results: &mut Vec<IVBootTimingSearchResult>,
    ) {
        // SIMD SHA-1計算
        let hashes = calculate_pokemon_sha1_simd(messages);

        // 各結果をチェック
        for i in 0..4 {
            let h0 = hashes[i * 5];
            let h1 = hashes[i * 5 + 1];
            let seed = calculate_pokemon_seed_from_hash(h0, h1);

            if self.target_seeds.contains(&seed) {
                let (seconds_since_2000, _, _) = batch_metadata[i];
                if let Some(result) = self.create_result(seed, h0, h1, seconds_since_2000) {
                    results.push(result);
                }
            }
        }
    }

    /// スカラー処理
    fn process_scalar(
        &self,
        message: &[u32; 16],
        seconds_since_2000: i64,
        _hardware_str: &str,
        results: &mut Vec<IVBootTimingSearchResult>,
    ) {
        let hash = crate::sha1::calculate_pokemon_sha1(message);
        let h0 = hash.0;
        let h1 = hash.1;
        let seed = calculate_pokemon_seed_from_hash(h0, h1);

        if self.target_seeds.contains(&seed) {
            if let Some(result) = self.create_result(seed, h0, h1, seconds_since_2000) {
                results.push(result);
            }
        }
    }

    /// 検索結果を生成
    fn create_result(
        &self,
        seed: u32,
        h0: u32,
        h1: u32,
        seconds_since_2000: i64,
    ) -> Option<IVBootTimingSearchResult> {
        let (year, month, day, hour, minute, second) =
            generate_display_datetime(seconds_since_2000)?;

        // LCG Seed計算 (h0, h1からリトルエンディアンで64bit値を構築)
        let (lcg_seed_high, lcg_seed_low) = calculate_lcg_seed_parts(h0, h1);

        Some(IVBootTimingSearchResult {
            mt_seed: seed,
            lcg_seed_high,
            lcg_seed_low,
            year,
            month,
            day,
            hour,
            minute,
            second,
            timer0: self.timer0,
            vcount: self.vcount,
            key_code: self.key_code,
        })
    }
}

/// SHA-1ハッシュ値から64bit LCG Seedのhigh/lowパートを計算
#[inline]
fn calculate_lcg_seed_parts(h0: u32, h1: u32) -> (u32, u32) {
    let h0_le = crate::sha1::swap_bytes_32(h0);
    let h1_le = crate::sha1::swap_bytes_32(h1);
    // LCG seed = (h1_le << 32) | h0_le
    // high = h1_le, low = h0_le
    (h1_le, h0_le)
}

// =============================================================================
// テスト
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_ds_config() -> DSConfigJs {
        let mac = [0x00, 0x09, 0xBF, 0xAA, 0xBB, 0xCC];
        let nazo = [0x02215F10, 0x02215F30, 0x02215F20, 0x02761008, 0x00000000];
        DSConfigJs::new(&mac, &nazo, "DS").unwrap()
    }

    fn create_test_segment() -> SegmentParamsJs {
        SegmentParamsJs::new(0x1000, 0x60, 0x2FFF)
    }

    fn create_test_time_range() -> TimeRangeParamsJs {
        TimeRangeParamsJs::new(0, 23, 0, 59, 0, 59).unwrap()
    }

    fn create_test_search_range() -> SearchRangeParamsJs {
        // 2024年1月1日から1時間
        SearchRangeParamsJs::new(2024, 1, 1, 3600).unwrap()
    }

    #[test]
    fn test_iterator_creation() {
        let ds_config = create_test_ds_config();
        let segment = create_test_segment();
        let time_range = create_test_time_range();
        let search_range = create_test_search_range();
        let target_seeds = [0x12345678u32];

        let iterator = IVBootTimingSearchIterator::new(
            &ds_config,
            &segment,
            &time_range,
            &search_range,
            &target_seeds,
        );

        assert!(iterator.is_ok());
        let iterator = iterator.unwrap();
        assert!(!iterator.is_finished());
        assert_eq!(iterator.total_seconds(), 3600);
        assert_eq!(iterator.processed_seconds(), 0);
    }

    #[test]
    fn test_iterator_empty_seeds() {
        let ds_config = create_test_ds_config();
        let segment = create_test_segment();
        let time_range = create_test_time_range();
        let search_range = create_test_search_range();
        let target_seeds: [u32; 0] = [];

        let result = IVBootTimingSearchIterator::new(
            &ds_config,
            &segment,
            &time_range,
            &search_range,
            &target_seeds,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_iterator_progress() {
        let ds_config = create_test_ds_config();
        let segment = create_test_segment();
        let time_range = create_test_time_range();
        let search_range = create_test_search_range();
        let target_seeds = [0x12345678u32];

        let mut iterator = IVBootTimingSearchIterator::new(
            &ds_config,
            &segment,
            &time_range,
            &search_range,
            &target_seeds,
        )
        .unwrap();

        // 最初の進捗は0
        assert_eq!(iterator.progress(), 0.0);

        // 100秒処理
        let _results = iterator.next_batch(100, 100);

        // 進捗が更新されている
        assert!(iterator.processed_seconds() > 0);
        assert!(iterator.progress() > 0.0);
    }

    #[test]
    fn test_iterator_completes() {
        let ds_config = create_test_ds_config();
        let segment = create_test_segment();
        let time_range = create_test_time_range();
        // 10秒だけ検索
        let search_range = SearchRangeParamsJs::new(2024, 1, 1, 10).unwrap();
        let target_seeds = [0x12345678u32];

        let mut iterator = IVBootTimingSearchIterator::new(
            &ds_config,
            &segment,
            &time_range,
            &search_range,
            &target_seeds,
        )
        .unwrap();

        // 全部処理
        while !iterator.is_finished() {
            let _results = iterator.next_batch(100, 100);
        }

        assert!(iterator.is_finished());
        assert_eq!(iterator.processed_seconds(), 10);
        assert_eq!(iterator.progress(), 1.0);
    }

    #[test]
    fn test_multiple_target_seeds() {
        let ds_config = create_test_ds_config();
        let segment = create_test_segment();
        let time_range = create_test_time_range();
        let search_range = create_test_search_range();
        // 複数のSeedを検索
        let target_seeds = [0x12345678u32, 0xABCDEF00u32, 0x11111111u32];

        let iterator = IVBootTimingSearchIterator::new(
            &ds_config,
            &segment,
            &time_range,
            &search_range,
            &target_seeds,
        );

        assert!(iterator.is_ok());
        let iterator = iterator.unwrap();
        assert_eq!(iterator.target_seeds.len(), 3);
    }
}

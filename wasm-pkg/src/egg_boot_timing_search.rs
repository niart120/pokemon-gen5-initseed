//! 孵化乱数起動時間検索器
//!
//! BW/BW2における孵化乱数（タマゴ生成）の起動時間検索機能を提供する。
//! 起動日時・SHA-1パラメータ・消費範囲・個体フィルター条件に基づき、
//! 条件に合致する個体とその起動条件を列挙する。
//!
//! ## WebGPU類似セグメントパターン設計
//!
//! WebGPU Seed検索と同様のセグメント分割パターンを採用:
//! 1. TypeScript側で timer0 × vcount × keyCode のセグメントループを実装
//! 2. `EggBootTimingSearchIterator::new()` で単一セグメント（固定timer0/vcount/keyCode）のイテレータを作成
//! 3. `EggBootTimingSearchIterator::next_batch()` で seconds 方向の結果をバッチ取得
//!    - result_limit件見つかるか、chunk_seconds秒分処理したら即座にreturn
//! 4. `is_finished` で完了判定
//!
//! ## 公開API
//! - `generate_egg_key_codes(key_input_mask)`: キーコード一覧を取得
//! - `EggBootTimingSearchIterator`: 単一セグメントの検索イテレータ

use crate::egg_iv::{
    Gender, GenerationConditions, GenerationConditionsJs, HiddenPowerInfo, IndividualFilter,
    IndividualFilterJs,
};
use crate::egg_seed_enumerator::{EggSeedEnumerator, ParentsIVs, ParentsIVsJs};
use crate::integrated_search::generate_key_codes;
use crate::offset_calculator::GameMode;
use crate::search_common::{
    build_ranged_time_code_table, BaseMessageBuilder, HashValuesEnumerator,
    DSConfigJs, SearchRangeParamsJs, SegmentParamsJs, TimeRangeParamsJs,
};
use wasm_bindgen::prelude::*;

/// キー入力マスクから有効なキーコード一覧を生成
///
/// TypeScript側でセグメントループを構築する際に使用。
/// 各キーコードに対して、対応するEggBootTimingSearchIteratorを作成する。
#[wasm_bindgen]
pub fn generate_egg_key_codes(key_input_mask: u32) -> Vec<u32> {
    generate_key_codes(key_input_mask)
}

/// 検索結果1件（起動条件 + 個体情報）
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct EggBootTimingSearchResult {
    // 起動条件
    year: u32,
    month: u32,
    date: u32,
    hour: u32,
    minute: u32,
    second: u32,
    timer0: u32,
    vcount: u32,
    key_code: u32,

    // LCG Seed
    lcg_seed_high: u32,
    lcg_seed_low: u32,

    // MT Seed (IV用)
    mt_seed: u32,

    // 個体情報
    advance: u64,
    is_stable: bool,
    ivs: [u8; 6],
    nature: u8,
    gender: u8,
    ability: u8,
    shiny: u8,
    pid: u32,
    hp_type: u8,
    hp_power: u8,
    hp_known: bool,
}

#[wasm_bindgen]
impl EggBootTimingSearchResult {
    #[wasm_bindgen(getter)]
    pub fn year(&self) -> u32 {
        self.year
    }

    #[wasm_bindgen(getter)]
    pub fn month(&self) -> u32 {
        self.month
    }

    #[wasm_bindgen(getter)]
    pub fn date(&self) -> u32 {
        self.date
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

    #[wasm_bindgen(getter = lcgSeedHex)]
    pub fn lcg_seed_hex(&self) -> String {
        let seed = ((self.lcg_seed_high as u64) << 32) | (self.lcg_seed_low as u64);
        format!("{seed:016X}")
    }

    #[wasm_bindgen(getter)]
    pub fn advance(&self) -> u64 {
        self.advance
    }

    #[wasm_bindgen(getter = isStable)]
    pub fn is_stable(&self) -> bool {
        self.is_stable
    }

    #[wasm_bindgen(getter)]
    pub fn ivs(&self) -> Vec<u8> {
        self.ivs.to_vec()
    }

    #[wasm_bindgen(getter)]
    pub fn nature(&self) -> u8 {
        self.nature
    }

    /// Gender: 0=Male, 1=Female, 2=Genderless
    #[wasm_bindgen(getter)]
    pub fn gender(&self) -> u8 {
        self.gender
    }

    /// Ability slot: 0=Ability1, 1=Ability2, 2=Hidden
    #[wasm_bindgen(getter)]
    pub fn ability(&self) -> u8 {
        self.ability
    }

    /// Shiny type: 0=Normal (not shiny), 1=Square shiny, 2=Star shiny
    #[wasm_bindgen(getter)]
    pub fn shiny(&self) -> u8 {
        self.shiny
    }

    #[wasm_bindgen(getter)]
    pub fn pid(&self) -> u32 {
        self.pid
    }

    #[wasm_bindgen(getter = hpType)]
    pub fn hp_type(&self) -> u8 {
        self.hp_type
    }

    #[wasm_bindgen(getter = hpPower)]
    pub fn hp_power(&self) -> u8 {
        self.hp_power
    }

    #[wasm_bindgen(getter = hpKnown)]
    pub fn hp_known(&self) -> bool {
        self.hp_known
    }

    #[wasm_bindgen(getter = mtSeed)]
    pub fn mt_seed(&self) -> u32 {
        self.mt_seed
    }

    #[wasm_bindgen(getter = mtSeedHex)]
    pub fn mt_seed_hex(&self) -> String {
        format!("{:08X}", self.mt_seed)
    }
}

/// 孵化乱数起動時間検索イテレータ
///
/// 単一セグメント（固定 timer0/vcount/keyCode）に対して seconds 方向の検索を行う。
/// TypeScript側で timer0 × vcount × keyCode のセグメントループを実装し、
/// 各セグメントに対してこのイテレータを作成する。
///
/// ## 使用例（TypeScript）
/// ```typescript
/// const keyCodes = wasm.generate_egg_key_codes(keyInputMask);
/// for (const timer0 of range(timer0Min, timer0Max)) {
///   for (const vcount of range(vcountMin, vcountMax)) {
///     for (const keyCode of keyCodes) {
///       const iterator = new EggBootTimingSearchIterator(..., timer0, vcount, keyCode);
///       while (!iterator.isFinished) {
///         const results = iterator.next_batch(256, 3600);
///         // 結果処理
///       }
///       iterator.free();
///     }
///   }
/// }
/// ```
#[wasm_bindgen]
pub struct EggBootTimingSearchIterator {
    // ハッシュ値列挙器（所有）
    hash_enumerator: HashValuesEnumerator,

    // セグメントパラメータ（結果出力用に保持）
    timer0: u32,
    vcount: u32,
    key_code: u32,

    // 孵化条件
    conditions: GenerationConditions,
    parents: ParentsIVs,
    filter: Option<IndividualFilter>,
    consider_npc_consumption: bool,
    game_mode: GameMode,
    user_offset: u64,
    advance_count: u32,

    // 検索範囲
    range_seconds: u32,

    // 現在位置
    current_offset: u32,
    finished: bool,
}

#[wasm_bindgen]
impl EggBootTimingSearchIterator {
    /// コンストラクタ
    ///
    /// # Arguments
    /// - `ds_config`: DS設定パラメータ (MAC/Nazo/Hardware)
    /// - `segment`: セグメントパラメータ (Timer0/VCount/KeyCode)
    /// - `time_range`: 時刻範囲パラメータ
    /// - `search_range`: 検索範囲パラメータ
    /// - `conditions`: 孵化条件
    /// - `parents`: 親個体値
    /// - `filter_js`: 個体フィルター（オプション）
    /// - `consider_npc_consumption`: NPC消費を考慮するか
    /// - `game_mode`: ゲームモード
    /// - `user_offset`: ユーザー指定オフセット
    /// - `advance_count`: 消費数
    #[wasm_bindgen(constructor)]
    pub fn new(
        ds_config: &DSConfigJs,
        segment: &SegmentParamsJs,
        time_range: &TimeRangeParamsJs,
        search_range: &SearchRangeParamsJs,
        conditions: &GenerationConditionsJs,
        parents: &ParentsIVsJs,
        filter_js: Option<IndividualFilterJs>,
        consider_npc_consumption: bool,
        game_mode: GameMode,
        user_offset: u64,
        advance_count: u32,
    ) -> Result<EggBootTimingSearchIterator, String> {
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
        let range_seconds = search_range.range_seconds();

        // HashValuesEnumerator構築
        let hash_enumerator =
            HashValuesEnumerator::new(base_message_builder, time_code_table, start_seconds, range_seconds);

        // 孵化条件の変換
        let internal_conditions = conditions.to_internal();
        let internal_parents = parents.to_internal();
        let internal_filter = filter_js.map(|f| f.to_internal());

        Ok(EggBootTimingSearchIterator {
            hash_enumerator,
            timer0: segment_internal.timer0,
            vcount: segment_internal.vcount,
            key_code: segment_internal.key_code,
            conditions: internal_conditions,
            parents: internal_parents,
            filter: internal_filter,
            consider_npc_consumption,
            game_mode,
            user_offset,
            advance_count,
            range_seconds,
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
    /// - result_limit件見つかったら即return
    /// - chunk_seconds秒分処理したら結果がなくても一旦return
    /// - 検索範囲を全て処理したらfinished=trueになる
    #[wasm_bindgen]
    pub fn next_batch(&mut self, result_limit: usize, chunk_seconds: u32) -> js_sys::Array {
        if self.finished {
            return js_sys::Array::new();
        }

        let mut results: Vec<EggBootTimingSearchResult> =
            Vec::with_capacity(std::cmp::min(result_limit, 256));

        let initial_processed = self.hash_enumerator.processed_seconds();
        let target_processed = initial_processed + chunk_seconds;

        // HashValuesEnumeratorからハッシュ値を4件ずつ取得して個体列挙
        loop {
            let (entries, len) = self.hash_enumerator.next_quad();
            if len == 0 {
                break;
            }

            // 同一バッチ内のエントリはすべて処理（境界での取りこぼし防止）
            for i in 0..len as usize {
                let entry = &entries[i];
                let lcg_seed = entry.hash.to_lcg_seed();
                let display = entry.datetime_code.to_display_datetime();

                self.enumerate_eggs_for_seed(lcg_seed, &display, &mut results, result_limit);
            }

            // result_limit到達チェック（バッチ処理完了後）
            if results.len() >= result_limit {
                break;
            }

            // チャンク処理制限
            if self.hash_enumerator.processed_seconds() >= target_processed {
                break;
            }
        }

        // 処理済み秒数を更新
        let current_processed = self.hash_enumerator.processed_seconds();
        self.current_offset = current_processed;

        // 検索完了チェック
        if current_processed >= self.range_seconds {
            self.finished = true;
        }

        self.to_js_array(results)
    }

    /// 結果をJS配列に変換
    fn to_js_array(&self, results: Vec<EggBootTimingSearchResult>) -> js_sys::Array {
        let js_array = js_sys::Array::new_with_length(results.len() as u32);
        for (i, result) in results.into_iter().enumerate() {
            js_array.set(i as u32, JsValue::from(result));
        }
        js_array
    }

    /// 指定されたLCG Seedに対して条件に合う個体を列挙
    ///
    /// EggSeedEnumeratorを使用して個体列挙ロジックを共通化。
    fn enumerate_eggs_for_seed(
        &self,
        lcg_seed: u64,
        display: &crate::search_common::DisplayDateTime,
        results: &mut Vec<EggBootTimingSearchResult>,
        max_results: usize,
    ) {
        let mut enumerator = EggSeedEnumerator::new(
            lcg_seed,
            self.user_offset,
            self.advance_count,
            self.conditions,
            self.parents,
            self.filter.clone(),
            self.consider_npc_consumption,
            self.game_mode,
        );

        while results.len() < max_results {
            match enumerator.next_egg() {
                Ok(Some(egg_data)) => {
                    let (hp_type, hp_power, hp_known) = match egg_data.egg.hidden_power {
                        HiddenPowerInfo::Known { r#type, power } => (r#type as u8, power, true),
                        HiddenPowerInfo::Unknown => (0, 0, false),
                    };

                    results.push(EggBootTimingSearchResult {
                        year: display.year,
                        month: display.month,
                        date: display.day,
                        hour: display.hour,
                        minute: display.minute,
                        second: display.second,
                        timer0: self.timer0,
                        vcount: self.vcount,
                        key_code: self.key_code,
                        lcg_seed_high: (lcg_seed >> 32) as u32,
                        lcg_seed_low: lcg_seed as u32,
                        mt_seed: egg_data.egg.mt_seed,
                        advance: egg_data.advance,
                        is_stable: egg_data.is_stable,
                        ivs: egg_data.egg.ivs,
                        nature: egg_data.egg.nature as u8,
                        gender: match egg_data.egg.gender {
                            Gender::Male => 0,
                            Gender::Female => 1,
                            Gender::Genderless => 2,
                        },
                        ability: egg_data.egg.ability as u8,
                        shiny: egg_data.egg.shiny as u8,
                        pid: egg_data.egg.pid,
                        hp_type,
                        hp_power,
                        hp_known,
                    });
                }
                Ok(None) => break,
                Err(_) => break,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search_common::{
        build_ranged_time_code_table, HardwareType, TimeRangeParams, EPOCH_2000_UNIX, SECONDS_PER_DAY,
    };
    use crate::sha1::swap_bytes_32;

    #[test]
    fn test_allowed_second_mask_basic() {
        // Build mask using TimeRangeParams
        let range = TimeRangeParams::new(10, 12, 0, 59, 0, 59).unwrap();
        let table = build_ranged_time_code_table(&range, HardwareType::DS);

        // 10:00:00 should be allowed
        let idx_10_00_00 = 10 * 3600;
        assert!(table[idx_10_00_00].is_some());

        // 09:59:59 should not be allowed
        let idx_09_59_59 = 9 * 3600 + 59 * 60 + 59;
        assert!(table[idx_09_59_59].is_none());

        // 13:00:00 should not be allowed
        let idx_13_00_00 = 13 * 3600;
        assert!(table[idx_13_00_00].is_none());
    }

    #[test]
    fn test_swap_bytes_32_consistency() {
        // Test swap_bytes_32 for LCG seed calculation
        let original: u32 = 0x12345678;
        let swapped = swap_bytes_32(original);
        // 0x12345678 -> 0x78563412
        assert_eq!(swapped, 0x78563412);

        // Double swap should return original
        let double_swapped = swap_bytes_32(swapped);
        assert_eq!(double_swapped, original);
    }

    #[test]
    fn test_epoch_constant() {
        // Verify 2000-01-01 00:00:00 UTC is correct
        use chrono::{NaiveDate, TimeZone, Utc};
        let epoch_2000 = NaiveDate::from_ymd_opt(2000, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        let epoch_2000_utc = Utc.from_utc_datetime(&epoch_2000);
        assert_eq!(epoch_2000_utc.timestamp(), EPOCH_2000_UNIX);
    }

    /// 時間範囲フィルタリングの統合テスト
    #[test]
    fn test_time_range_filtering() {
        // 12:00-14:00の時間範囲でテーブルを作成
        let range = TimeRangeParams::new(12, 14, 0, 59, 0, 59).unwrap();
        let table = build_ranged_time_code_table(&range, HardwareType::DS);

        // 範囲内の時刻は許可される
        assert!(table[12 * 3600].is_some()); // 12:00:00
        assert!(table[13 * 3600 + 30 * 60].is_some()); // 13:30:00
        assert!(table[14 * 3600 + 59 * 60 + 59].is_some()); // 14:59:59

        // 範囲外の時刻は許可されない
        assert!(table[11 * 3600 + 59 * 60 + 59].is_none()); // 11:59:59
        assert!(table[15 * 3600].is_none()); // 15:00:00

        // 7日間の検索で、許可される秒数を計算
        let allowed_seconds_per_day: usize = table.iter().filter(|t| t.is_some()).count();
        assert_eq!(allowed_seconds_per_day, 3 * 60 * 60); // 3時間 = 10800秒
        
        // 7日間で許可される総秒数
        let total_allowed_in_7_days = allowed_seconds_per_day * 7;
        assert_eq!(total_allowed_in_7_days, 75600);
    }

    /// キーコード生成のテスト
    #[test]
    fn test_key_code_generation_for_search() {
        // キー入力なし
        let codes_no_key = generate_key_codes(0);
        assert!(!codes_no_key.is_empty());

        // Aボタンのみ
        let codes_a_only = generate_key_codes(0x0001);
        assert!(codes_a_only.len() >= 2); // Aなし + Aあり

        // 複数キー
        let codes_multi = generate_key_codes(0x0003); // A + B
        assert!(codes_multi.len() >= 4); // 2^2の組み合わせ
    }

    /// 時刻範囲マスクのテスト（分・秒に制約がある場合）
    #[test]
    fn test_allowed_second_mask_with_minute_second_constraints() {
        // hour: 10-11, minute: 30-45, second: 0-30 の場合
        let range = TimeRangeParams::new(10, 11, 30, 45, 0, 30).unwrap();
        let table = build_ranged_time_code_table(&range, HardwareType::DS);

        // 10:30:00 は許可される
        let idx_10_30_00 = 10 * 3600 + 30 * 60;
        assert!(table[idx_10_30_00].is_some(), "10:30:00 should be allowed");

        // 10:30:30 は許可される
        let idx_10_30_30 = 10 * 3600 + 30 * 60 + 30;
        assert!(table[idx_10_30_30].is_some(), "10:30:30 should be allowed");

        // 10:30:31 は許可されない（秒が範囲外）
        let idx_10_30_31 = 10 * 3600 + 30 * 60 + 31;
        assert!(table[idx_10_30_31].is_none(), "10:30:31 should NOT be allowed");

        // 10:29:00 は許可されない（分が範囲外）
        let idx_10_29_00 = 10 * 3600 + 29 * 60;
        assert!(table[idx_10_29_00].is_none(), "10:29:00 should NOT be allowed");

        // 11:00:00 は許可されない（分が範囲外：0分は30-45の範囲外）
        let idx_11_00_00 = 11 * 3600;
        assert!(table[idx_11_00_00].is_none(), "11:00:00 should NOT be allowed (minute 0 is outside 30-45)");

        // 11:30:00 は許可される
        let idx_11_30_00 = 11 * 3600 + 30 * 60;
        assert!(table[idx_11_30_00].is_some(), "11:30:00 should be allowed");

        // 11:45:30 は許可される
        let idx_11_45_30 = 11 * 3600 + 45 * 60 + 30;
        assert!(table[idx_11_45_30].is_some(), "11:45:30 should be allowed");

        // 許可される秒数を計算
        // hour: 2時間 (10, 11)
        // minute: 16分 (30-45)
        // second: 31秒 (0-30)
        // => 2 * 16 * 31 = 992秒
        let allowed_count: usize = table.iter().filter(|t| t.is_some()).count();
        assert_eq!(allowed_count, 2 * 16 * 31, "Expected 992 allowed seconds");
    }

    /// 全範囲許可のテスト
    #[test]
    fn test_allowed_second_mask_full_range() {
        let range = TimeRangeParams::new(0, 23, 0, 59, 0, 59).unwrap();
        let table = build_ranged_time_code_table(&range, HardwareType::DS);

        // 全86400秒が許可される
        let allowed_count: usize = table.iter().filter(|t| t.is_some()).count();
        assert_eq!(allowed_count, 86400, "All seconds should be allowed");
    }

    /// 日時復元のテスト（generate_display_datetime相当のロジック）
    #[test]
    fn test_datetime_restoration_from_seconds_since_2000() {
        use chrono::{DateTime, Datelike, NaiveDate, Timelike};
        
        // 2025-01-15 10:30:45 を復元するテスト
        let target_datetime = NaiveDate::from_ymd_opt(2025, 1, 15)
            .unwrap()
            .and_hms_opt(10, 30, 45)
            .unwrap();
        let target_unix = target_datetime.and_utc().timestamp();
        let seconds_since_2000 = target_unix - EPOCH_2000_UNIX;
        
        // 復元
        let restored = DateTime::from_timestamp(seconds_since_2000 + EPOCH_2000_UNIX, 0)
            .unwrap()
            .naive_utc();
        
        assert_eq!(restored.year(), 2025);
        assert_eq!(restored.month(), 1);
        assert_eq!(restored.day(), 15);
        assert_eq!(restored.hour(), 10);
        assert_eq!(restored.minute(), 30);
        assert_eq!(restored.second(), 45);
    }

    /// 複数日にまたがる場合の日時復元テスト
    #[test]
    fn test_datetime_restoration_multi_day() {
        use chrono::{DateTime, Datelike, NaiveDate, Timelike};
        
        // 開始日: 2025-01-15 00:00:00
        let start_date = NaiveDate::from_ymd_opt(2025, 1, 15)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        let start_unix = start_date.and_utc().timestamp();
        let base_seconds_since_2000 = start_unix - EPOCH_2000_UNIX;
        
        // 2日目の 10:30:00 をテスト
        // second_offset = 1日分 + 10時間30分 = 86400 + 37800 = 124200
        let second_offset = 86400 + 10 * 3600 + 30 * 60;
        let current_seconds = base_seconds_since_2000 + second_offset as i64;
        
        let restored = DateTime::from_timestamp(current_seconds + EPOCH_2000_UNIX, 0)
            .unwrap()
            .naive_utc();
        
        // 2025-01-16 10:30:00 になるはず
        assert_eq!(restored.year(), 2025);
        assert_eq!(restored.month(), 1);
        assert_eq!(restored.day(), 16, "Should be day 16 (2nd day)");
        assert_eq!(restored.hour(), 10);
        assert_eq!(restored.minute(), 30);
        assert_eq!(restored.second(), 0);
    }

    /// second_of_day 計算のテスト（複数日にまたがる場合）
    #[test]
    fn test_second_of_day_calculation_multi_day() {
        use chrono::NaiveDate;
        
        // 開始日: 2025-01-15 00:00:00
        let start_date = NaiveDate::from_ymd_opt(2025, 1, 15)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        let start_unix = start_date.and_utc().timestamp();
        let base_seconds_since_2000 = start_unix - EPOCH_2000_UNIX;
        
        // 1日目の 10:30:00
        let offset_day1 = 10 * 3600 + 30 * 60; // 37800
        let current_day1 = base_seconds_since_2000 + offset_day1 as i64;
        let second_of_day_1 = (current_day1 % SECONDS_PER_DAY) as u32;
        assert_eq!(second_of_day_1, 37800, "Day 1 10:30:00 should be 37800 seconds");
        
        // 2日目の 10:30:00
        let offset_day2 = 86400 + 10 * 3600 + 30 * 60; // 124200
        let current_day2 = base_seconds_since_2000 + offset_day2 as i64;
        let second_of_day_2 = (current_day2 % SECONDS_PER_DAY) as u32;
        assert_eq!(second_of_day_2, 37800, "Day 2 10:30:00 should also be 37800 seconds");
        
        // 両日とも同じ second_of_day になるはず
        assert_eq!(second_of_day_1, second_of_day_2);
    }

    /// 実際のユースケース: 2025/11/26-29, h:00-23, m:00-59, s:11-11
    /// 日付範囲外の結果が返らないことを確認
    #[test]
    fn test_range_seconds_calculation_4_days() {
        use chrono::{DateTime, Datelike, NaiveDate};
        
        // 開始日: 2025-11-26 00:00:00
        let start_date = NaiveDate::from_ymd_opt(2025, 11, 26)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        let start_unix = start_date.and_utc().timestamp();
        let base_seconds_since_2000 = start_unix - EPOCH_2000_UNIX;
        
        // 4日間 = 2025/11/26, 27, 28, 29
        let total_days = 4u32;
        let range_seconds = total_days * 86400;
        
        // 最後の許可される秒: range_seconds - 1
        let last_allowed_offset = range_seconds - 1;
        let last_current_seconds = base_seconds_since_2000 + last_allowed_offset as i64;
        
        let last_datetime = DateTime::from_timestamp(last_current_seconds + EPOCH_2000_UNIX, 0)
            .unwrap()
            .naive_utc();
        
        // 最後の許可される秒は 2025-11-29 23:59:59 であるべき
        assert_eq!(last_datetime.year(), 2025, "Last year should be 2025");
        assert_eq!(last_datetime.month(), 11, "Last month should be 11");
        assert_eq!(last_datetime.day(), 29, "Last day should be 29");
        
        // range_seconds の次の秒（許可されない）
        let first_invalid_offset = range_seconds;
        let first_invalid_seconds = base_seconds_since_2000 + first_invalid_offset as i64;
        
        let first_invalid_datetime = DateTime::from_timestamp(first_invalid_seconds + EPOCH_2000_UNIX, 0)
            .unwrap()
            .naive_utc();
        
        // 許可されない最初の秒は 2025-11-30 00:00:00 であるべき
        assert_eq!(first_invalid_datetime.year(), 2025, "Invalid year should be 2025");
        assert_eq!(first_invalid_datetime.month(), 11, "Invalid month should be 11");
        assert_eq!(first_invalid_datetime.day(), 30, "Invalid day should be 30 (out of range)");
    }

    /// second_offset が range_seconds 未満で正しく制限されることを確認
    #[test]
    fn test_second_offset_boundary() {
        // 4日間のシミュレーション
        let range_seconds: u32 = 4 * 86400; // 345600
        
        // ループ条件のテスト
        let mut second_offset: u32 = 0;
        let mut max_offset_reached: u32 = 0;
        
        while second_offset < range_seconds {
            max_offset_reached = second_offset;
            second_offset += 1;
            
            // 途中でブレイク（実際のループをシミュレート）
            if second_offset > 345600 {
                panic!("second_offset exceeded range_seconds!");
            }
        }
        
        // 最大オフセットは range_seconds - 1
        assert_eq!(max_offset_reached, range_seconds - 1);
        assert_eq!(max_offset_reached, 345599);
    }
}

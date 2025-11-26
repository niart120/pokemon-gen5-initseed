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

use crate::datetime_codes::{DateCodeGenerator, TimeCodeGenerator};
use crate::egg_iv::{
    Gender, GenerationConditions, GenerationConditionsJs, HiddenPowerInfo, IndividualFilter,
    IndividualFilterJs,
};
use crate::egg_seed_enumerator::{EggSeedEnumerator, ParentsIVs, ParentsIVsJs};
use crate::integrated_search::generate_key_codes;
use crate::offset_calculator::GameMode;
use crate::sha1::{calculate_pokemon_sha1, swap_bytes_32};
use crate::sha1_simd::calculate_pokemon_sha1_simd;
use chrono::{Datelike, NaiveDate, Timelike};
use wasm_bindgen::prelude::*;

/// 2000年1月1日 00:00:00 UTCのUnix時間
const EPOCH_2000_UNIX: i64 = 946684800;
const SECONDS_PER_DAY: i64 = 86_400;

/// Hardware別のframe値
const HARDWARE_FRAME_DS: u32 = 8;
const HARDWARE_FRAME_DS_LITE: u32 = 6;
const HARDWARE_FRAME_3DS: u32 = 9;

/// Hardwareからframe値を取得
fn get_frame_for_hardware(hardware: &str) -> Result<u32, JsValue> {
    match hardware {
        "DS" => Ok(HARDWARE_FRAME_DS),
        "DS_LITE" => Ok(HARDWARE_FRAME_DS_LITE),
        "3DS" => Ok(HARDWARE_FRAME_3DS),
        _ => Err(JsValue::from_str("Hardware must be DS, DS_LITE, or 3DS")),
    }
}

/// キー入力マスクから有効なキーコード一覧を生成
///
/// TypeScript側でセグメントループを構築する際に使用。
/// 各キーコードに対して、対応するEggBootTimingSearchIteratorを作成する。
#[wasm_bindgen]
pub fn generate_egg_key_codes(key_input_mask: u32) -> Vec<u32> {
    generate_key_codes(key_input_mask)
}

/// 日時範囲設定
#[derive(Clone, Copy)]
struct DailyTimeRangeConfig {
    hour_start: u32,
    hour_end: u32,
    minute_start: u32,
    minute_end: u32,
    second_start: u32,
    second_end: u32,
}

impl DailyTimeRangeConfig {
    fn new(
        hour_start: u32,
        hour_end: u32,
        minute_start: u32,
        minute_end: u32,
        second_start: u32,
        second_end: u32,
    ) -> Result<Self, JsValue> {
        fn validate(
            label: &str,
            start: u32,
            end: u32,
            min: u32,
            max: u32,
        ) -> Result<(u32, u32), JsValue> {
            if start < min || end > max {
                return Err(JsValue::from_str(&format!(
                    "{label} range must be within {min}..={max}",
                )));
            }
            if start > end {
                return Err(JsValue::from_str(&format!(
                    "{label} range start ({start}) must be <= end ({end})",
                )));
            }
            Ok((start, end))
        }

        let (hour_start, hour_end) = validate("hour", hour_start, hour_end, 0, 23)?;
        let (minute_start, minute_end) = validate("minute", minute_start, minute_end, 0, 59)?;
        let (second_start, second_end) = validate("second", second_start, second_end, 0, 59)?;

        Ok(DailyTimeRangeConfig {
            hour_start,
            hour_end,
            minute_start,
            minute_end,
            second_start,
            second_end,
        })
    }
}

fn build_allowed_second_mask(range: &DailyTimeRangeConfig) -> Box<[bool; 86400]> {
    let mut mask = Box::new([false; 86400]);
    for hour in range.hour_start..=range.hour_end {
        for minute in range.minute_start..=range.minute_end {
            for second in range.second_start..=range.second_end {
                let idx = (hour * 3600 + minute * 60 + second) as usize;
                mask[idx] = true;
            }
        }
    }
    mask
}

/// SHA-1ハッシュ値から64bit LCG Seedを計算
#[inline]
fn calculate_lcg_seed_from_hash(h0: u32, h1: u32) -> u64 {
    let h0_le = swap_bytes_32(h0) as u64;
    let h1_le = swap_bytes_32(h1) as u64;
    (h1_le << 32) | h0_le
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
        format!("{:016X}", seed)
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
    // 検索パラメータ
    hardware: String,
    base_message: [u32; 16],
    allowed_second_mask: Box<[bool; 86400]>,
    conditions: GenerationConditions,
    parents: ParentsIVs,
    filter: Option<IndividualFilter>,
    consider_npc_consumption: bool,
    game_mode: GameMode,
    user_offset: u64,
    advance_count: u32,

    // 固定セグメントパラメータ
    timer0: u32,
    vcount: u32,
    key_code: u32,

    // 検索範囲
    base_seconds_since_2000: i64,
    range_seconds: u32,

    // イテレータ状態（1次元: secondsのみ）
    second_offset: u32,
    finished: bool,
}

#[wasm_bindgen]
impl EggBootTimingSearchIterator {
    /// コンストラクタ
    ///
    /// 単一セグメント（固定 timer0/vcount/keyCode）のイテレータを作成。
    /// frame は hardware から自動導出される。
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        // SHA-1 パラメータ
        mac: &[u8],
        nazo: &[u32],
        hardware: &str,
        // 固定セグメント値
        timer0: u32,
        vcount: u32,
        key_code: u32,
        // 時刻範囲（日次フィルタ）
        hour_start: u32,
        hour_end: u32,
        minute_start: u32,
        minute_end: u32,
        second_start: u32,
        second_end: u32,
        // 検索開始日時
        year_start: u32,
        month_start: u32,
        date_start: u32,
        range_seconds: u32,
        // 孵化条件
        conditions: &GenerationConditionsJs,
        parents: &ParentsIVsJs,
        filter_js: Option<IndividualFilterJs>,
        consider_npc_consumption: bool,
        game_mode: GameMode,
        // 消費範囲
        user_offset: u64,
        advance_count: u32,
    ) -> Result<EggBootTimingSearchIterator, JsValue> {
        // バリデーション
        if mac.len() != 6 {
            return Err(JsValue::from_str("MAC address must be 6 bytes"));
        }
        if nazo.len() != 5 {
            return Err(JsValue::from_str("nazo must be 5 32-bit words"));
        }

        // hardwareからframe値を取得
        let frame = get_frame_for_hardware(hardware)?;

        let time_range = DailyTimeRangeConfig::new(
            hour_start,
            hour_end,
            minute_start,
            minute_end,
            second_start,
            second_end,
        )?;
        let allowed_second_mask = build_allowed_second_mask(&time_range);

        // 開始日時をUnix時間に変換
        let start_datetime =
            match NaiveDate::from_ymd_opt(year_start as i32, month_start, date_start)
                .and_then(|date| date.and_hms_opt(0, 0, 0))
            {
                Some(datetime) => datetime,
                None => return Err(JsValue::from_str("Invalid start datetime")),
            };
        let start_unix = start_datetime.and_utc().timestamp();
        let base_seconds_since_2000 = start_unix - EPOCH_2000_UNIX;

        // 基本メッセージテンプレートを事前構築
        let mut base_message = [0u32; 16];

        // data[0-4]: Nazo values
        for i in 0..5 {
            base_message[i] = swap_bytes_32(nazo[i]);
        }

        // data[6]: MAC address lower 16 bits
        let mac_lower = ((mac[4] as u32) << 8) | (mac[5] as u32);
        base_message[6] = mac_lower;

        // data[7]: MAC address upper 32 bits XOR GxStat XOR Frame
        let mac_upper = (mac[0] as u32)
            | ((mac[1] as u32) << 8)
            | ((mac[2] as u32) << 16)
            | ((mac[3] as u32) << 24);
        let gx_stat = 0x06000000u32;
        let data7 = mac_upper ^ gx_stat ^ frame;
        base_message[7] = swap_bytes_32(data7);

        // Fixed values
        base_message[10] = 0x00000000;
        base_message[11] = 0x00000000;
        base_message[12] = 0;
        base_message[13] = 0x80000000;
        base_message[14] = 0x00000000;
        base_message[15] = 0x000001A0;

        // 孵化条件の変換
        let internal_conditions = conditions.to_internal();
        let internal_parents = parents.to_internal();
        let internal_filter = filter_js.map(|f| f.to_internal());

        Ok(EggBootTimingSearchIterator {
            hardware: hardware.to_string(),
            base_message,
            allowed_second_mask,
            conditions: internal_conditions,
            parents: internal_parents,
            filter: internal_filter,
            consider_npc_consumption,
            game_mode,
            user_offset,
            advance_count,
            timer0,
            vcount,
            key_code,
            base_seconds_since_2000,
            range_seconds,
            second_offset: 0,
            finished: false,
        })
    }

    /// 検索が完了したかどうか
    #[wasm_bindgen(getter = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.finished
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

        let mut seconds_processed: u32 = 0;

        // SIMD バッチ処理用バッファ
        let mut messages = [0u32; 64];
        let mut batch_metadata: [i64; 4] = [0; 4];
        let mut batch_len = 0usize;

        while self.second_offset < self.range_seconds {
            // 終了条件チェック
            if results.len() >= result_limit {
                self.flush_simd_batch(
                    &messages,
                    &batch_metadata,
                    batch_len,
                    &mut results,
                    result_limit,
                );
                return self.to_js_array(results);
            }

            if seconds_processed >= chunk_seconds {
                self.flush_simd_batch(
                    &messages,
                    &batch_metadata,
                    batch_len,
                    &mut results,
                    result_limit,
                );
                return self.to_js_array(results);
            }

            let current_seconds = self.base_seconds_since_2000 + self.second_offset as i64;

            // 日時コード計算（許可範囲チェック含む）
            if let Some((time_code, date_code)) = self.calculate_datetime_codes(current_seconds) {
                // メッセージ構築
                let message = self.build_message(date_code, time_code);
                let base_idx = batch_len * 16;
                messages[base_idx..base_idx + 16].copy_from_slice(&message);
                batch_metadata[batch_len] = current_seconds;
                batch_len += 1;

                // 4件溜まったらSIMD処理
                if batch_len == 4 {
                    self.process_simd_batch(
                        &messages,
                        &batch_metadata,
                        batch_len,
                        &mut results,
                        result_limit,
                    );
                    batch_len = 0;
                }
            }

            self.second_offset += 1;
            seconds_processed += 1;
        }

        // 残りを処理
        if batch_len > 0 {
            self.process_simd_batch(
                &messages,
                &batch_metadata,
                batch_len,
                &mut results,
                result_limit,
            );
        }

        // 全範囲を処理完了
        self.finished = true;
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

    /// SIMDバッチ処理（結果件数制限チェックなし、即flush用）
    #[inline]
    fn flush_simd_batch(
        &self,
        messages: &[u32; 64],
        batch_metadata: &[i64; 4],
        batch_size: usize,
        results: &mut Vec<EggBootTimingSearchResult>,
        max_results: usize,
    ) {
        if batch_size > 0 {
            self.process_simd_batch(messages, batch_metadata, batch_size, results, max_results);
        }
    }

    /// SIMDバッチ処理
    #[inline]
    fn process_simd_batch(
        &self,
        messages: &[u32; 64],
        batch_metadata: &[i64; 4],
        batch_size: usize,
        results: &mut Vec<EggBootTimingSearchResult>,
        max_results: usize,
    ) {
        if batch_size == 0 {
            return;
        }

        // SIMD または スカラー SHA-1 計算
        let hash_results = if batch_size == 4 {
            calculate_pokemon_sha1_simd(messages)
        } else {
            // スカラーフォールバック
            let mut scalar_results = [0u32; 20];
            for i in 0..batch_size {
                let mut single_message = [0u32; 16];
                let base_idx = i * 16;
                single_message.copy_from_slice(&messages[base_idx..base_idx + 16]);
                let (h0, h1, h2, h3, h4) = calculate_pokemon_sha1(&single_message);
                scalar_results[i * 5] = h0;
                scalar_results[i * 5 + 1] = h1;
                scalar_results[i * 5 + 2] = h2;
                scalar_results[i * 5 + 3] = h3;
                scalar_results[i * 5 + 4] = h4;
            }
            scalar_results
        };

        // 各LCG Seedに対して個体検索
        for i in 0..batch_size {
            if results.len() >= max_results {
                return;
            }

            let h0 = hash_results[i * 5];
            let h1 = hash_results[i * 5 + 1];
            let lcg_seed = calculate_lcg_seed_from_hash(h0, h1);

            let current_seconds = batch_metadata[i];

            // 日時情報を取得
            if let Some(datetime) = self.generate_display_datetime(current_seconds) {
                self.enumerate_eggs_for_seed(lcg_seed, datetime, results, max_results);
            }
        }
    }

    /// 指定されたLCG Seedに対して条件に合う個体を列挙
    ///
    /// EggSeedEnumeratorを使用して個体列挙ロジックを共通化。
    fn enumerate_eggs_for_seed(
        &self,
        lcg_seed: u64,
        datetime: (u32, u32, u32, u32, u32, u32),
        results: &mut Vec<EggBootTimingSearchResult>,
        max_results: usize,
    ) {
        let (year, month, date, hour, minute, second) = datetime;

        let mut enumerator = EggSeedEnumerator::new(
            lcg_seed,
            self.user_offset,
            self.advance_count,
            self.conditions.clone(),
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
                        year,
                        month,
                        date,
                        hour,
                        minute,
                        second,
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

    /// 日時コード生成
    #[inline(always)]
    fn calculate_datetime_codes(&self, seconds_since_2000: i64) -> Option<(u32, u32)> {
        if seconds_since_2000 < 0 {
            return None;
        }

        let seconds_of_day = (seconds_since_2000 % SECONDS_PER_DAY) as u32;
        if !self.is_second_allowed(seconds_of_day) {
            return None;
        }

        let date_index = (seconds_since_2000 / SECONDS_PER_DAY) as u32;

        let time_code =
            TimeCodeGenerator::get_time_code_for_hardware(seconds_of_day, &self.hardware);
        let date_code = DateCodeGenerator::get_date_code(date_index);

        Some((time_code, date_code))
    }

    #[inline(always)]
    fn is_second_allowed(&self, second_of_day: u32) -> bool {
        self.allowed_second_mask[second_of_day as usize]
    }

    /// 結果表示用の日時を生成
    fn generate_display_datetime(
        &self,
        seconds_since_2000: i64,
    ) -> Option<(u32, u32, u32, u32, u32, u32)> {
        let result_datetime =
            chrono::DateTime::from_timestamp(seconds_since_2000 + EPOCH_2000_UNIX, 0)?.naive_utc();

        Some((
            result_datetime.year() as u32,
            result_datetime.month(),
            result_datetime.day(),
            result_datetime.hour(),
            result_datetime.minute(),
            result_datetime.second(),
        ))
    }

    /// メッセージ構築
    #[inline(always)]
    fn build_message(&self, date_code: u32, time_code: u32) -> [u32; 16] {
        let mut message = self.base_message;
        message[5] = swap_bytes_32((self.vcount << 16) | self.timer0);
        message[8] = date_code;
        message[9] = time_code;
        message[12] = swap_bytes_32(self.key_code);
        message
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcg_seed_calculation() {
        // h0=0x12345678, h1=0xABCDEF01 の場合
        let h0: u32 = 0x12345678;
        let h1: u32 = 0xABCDEF01;
        let seed = calculate_lcg_seed_from_hash(h0, h1);

        // swap_bytes_32の結果を確認
        let h0_le = swap_bytes_32(h0);
        let h1_le = swap_bytes_32(h1);
        let expected = ((h1_le as u64) << 32) | (h0_le as u64);
        assert_eq!(seed, expected);
    }

    #[test]
    fn test_allowed_second_mask_basic() {
        // Build mask manually with known valid range
        let mut mask = Box::new([false; 86400]);
        // Allow hours 10-12, all minutes and seconds
        for hour in 10..=12 {
            for minute in 0..=59 {
                for second in 0..=59 {
                    let idx = (hour * 3600 + minute * 60 + second) as usize;
                    mask[idx] = true;
                }
            }
        }

        // 10:00:00 should be allowed
        let idx_10_00_00 = 10 * 3600 + 0 * 60 + 0;
        assert!(mask[idx_10_00_00]);

        // 09:59:59 should not be allowed
        let idx_09_59_59 = 9 * 3600 + 59 * 60 + 59;
        assert!(!mask[idx_09_59_59]);

        // 13:00:00 should not be allowed
        let idx_13_00_00 = 13 * 3600 + 0 * 60 + 0;
        assert!(!mask[idx_13_00_00]);
    }

    #[test]
    fn test_lcg_seed_from_known_hash() {
        // Test with known values
        let h0: u32 = 0;
        let h1: u32 = 0;
        let seed = calculate_lcg_seed_from_hash(h0, h1);
        assert_eq!(seed, 0);

        // Test with 0xFFFFFFFF values
        let h0: u32 = 0xFFFFFFFF;
        let h1: u32 = 0xFFFFFFFF;
        let seed = calculate_lcg_seed_from_hash(h0, h1);
        assert_eq!(seed, 0xFFFFFFFFFFFFFFFF);
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

    // =========================================================================
    // 実検索機能テスト（7日間検索 + フィルター適用）
    // =========================================================================

    /// SHA-1とLCG Seed計算の統合テスト
    #[test]
    fn test_sha1_and_lcg_seed_integration() {
        // 固定メッセージでSHA-1計算をテスト
        let mut message = [0u32; 16];
        // Nazo values (Black Japanese version)
        message[0] = swap_bytes_32(0x02215F10);
        message[1] = swap_bytes_32(0x0221600C);
        message[2] = swap_bytes_32(0x0221600C);
        message[3] = swap_bytes_32(0x02216058);
        message[4] = swap_bytes_32(0x02216058);
        // VCount | Timer0
        message[5] = swap_bytes_32((0x60 << 16) | 0x0C79);
        // MAC lower
        message[6] = (0x34 << 8) | 0x56;
        // MAC upper XOR GxStat XOR Frame
        let mac_upper = 0x00 | (0x09 << 8) | (0xBF << 16) | (0x12 << 24);
        let gx_stat = 0x06000000u32;
        message[7] = swap_bytes_32(mac_upper ^ gx_stat ^ 8);
        // Date/Time codes (2025-01-15 12:00:00)
        message[8] = 0; // date code placeholder
        message[9] = 0; // time code placeholder
        message[10] = 0x00000000;
        message[11] = 0x00000000;
        message[12] = 0; // key code
        message[13] = 0x80000000;
        message[14] = 0x00000000;
        message[15] = 0x000001A0;

        let (h0, h1, h2, h3, h4) = calculate_pokemon_sha1(&message);

        // SHA-1結果が確定的であることを確認
        assert_ne!(h0, 0);
        assert_ne!(h1, 0);
        
        // LCG Seed計算
        let lcg_seed = calculate_lcg_seed_from_hash(h0, h1);
        
        // LCG Seedが非ゼロであることを確認
        assert_ne!(lcg_seed, 0);

        // 同じ入力で同じ結果が得られることを確認
        let (h0_2, h1_2, _, _, _) = calculate_pokemon_sha1(&message);
        let lcg_seed_2 = calculate_lcg_seed_from_hash(h0_2, h1_2);
        assert_eq!(lcg_seed, lcg_seed_2);

        // h2, h3, h4 も検証に含める
        assert_eq!(h2, calculate_pokemon_sha1(&message).2);
        assert_eq!(h3, calculate_pokemon_sha1(&message).3);
        assert_eq!(h4, calculate_pokemon_sha1(&message).4);
    }

    /// SIMD SHA-1 バッチ処理の統合テスト
    #[test]
    fn test_simd_batch_sha1_integration() {
        // 4つの異なるメッセージを準備
        let mut messages = [0u32; 64];
        
        for batch_idx in 0..4 {
            let base_idx = batch_idx * 16;
            
            // 各バッチに異なるTimer0値を設定
            let timer0 = 0x0C79 + batch_idx as u32;
            
            messages[base_idx] = swap_bytes_32(0x02215F10);
            messages[base_idx + 1] = swap_bytes_32(0x0221600C);
            messages[base_idx + 2] = swap_bytes_32(0x0221600C);
            messages[base_idx + 3] = swap_bytes_32(0x02216058);
            messages[base_idx + 4] = swap_bytes_32(0x02216058);
            messages[base_idx + 5] = swap_bytes_32((0x60 << 16) | timer0);
            messages[base_idx + 6] = (0x34 << 8) | 0x56;
            messages[base_idx + 7] = swap_bytes_32(0x00_09_BF_12 ^ 0x06000000 ^ 8);
            messages[base_idx + 13] = 0x80000000;
            messages[base_idx + 15] = 0x000001A0;
        }

        let results = calculate_pokemon_sha1_simd(&messages);

        // 4組の結果が得られることを確認
        assert_eq!(results.len(), 20);

        // 各バッチの結果が異なることを確認（Timer0が異なるため）
        let seeds: Vec<u64> = (0..4)
            .map(|i| calculate_lcg_seed_from_hash(results[i * 5], results[i * 5 + 1]))
            .collect();

        // 異なるTimer0値なので、少なくとも一部は異なるはず
        let unique_count = seeds.iter().collect::<std::collections::HashSet<_>>().len();
        assert!(unique_count >= 2, "Expected different seeds for different Timer0 values");
    }

    /// 日時コード生成の範囲テスト（7日間相当）
    #[test]
    fn test_datetime_code_generation_7_days() {
        // 7日間 = 604800秒
        let seconds_in_7_days = 7 * 24 * 60 * 60;
        
        // サンプリングしてテスト（全秒をテストすると遅すぎる）
        let sample_points = [0, 1000, 10000, 100000, 300000, 500000, 604799];
        
        for seconds in sample_points {
            if seconds >= seconds_in_7_days {
                continue;
            }
            
            let date_index = seconds / SECONDS_PER_DAY as u32;
            let seconds_of_day = seconds % SECONDS_PER_DAY as u32;
            
            let date_code = DateCodeGenerator::get_date_code(date_index);
            let time_code = TimeCodeGenerator::get_time_code_for_hardware(seconds_of_day, "DS");
            
            // コードが有効な範囲であることを確認
            assert_ne!(date_code, 0, "Date code should not be 0 for date_index={}", date_index);
            // time_codeは0の可能性があるのでチェックしない
            let _ = time_code; // 使用していることを明示
        }
    }

    /// 時間範囲フィルタリングの統合テスト
    #[test]
    fn test_time_range_filtering() {
        // 12:00-14:00の時間範囲でマスクを作成
        let mut mask = Box::new([false; 86400]);
        for hour in 12..=14 {
            for minute in 0..=59 {
                for second in 0..=59 {
                    let idx = (hour * 3600 + minute * 60 + second) as usize;
                    mask[idx] = true;
                }
            }
        }

        // 範囲内の時刻は許可される
        assert!(mask[12 * 3600]); // 12:00:00
        assert!(mask[13 * 3600 + 30 * 60]); // 13:30:00
        assert!(mask[14 * 3600 + 59 * 60 + 59]); // 14:59:59

        // 範囲外の時刻は許可されない
        assert!(!mask[11 * 3600 + 59 * 60 + 59]); // 11:59:59
        assert!(!mask[15 * 3600]); // 15:00:00

        // 7日間の検索で、許可される秒数を計算
        let allowed_seconds_per_day: usize = mask.iter().filter(|&&b| b).count();
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
        let range = DailyTimeRangeConfig::new(10, 11, 30, 45, 0, 30).unwrap();
        let mask = build_allowed_second_mask(&range);

        // 10:30:00 は許可される
        let idx_10_30_00 = 10 * 3600 + 30 * 60 + 0;
        assert!(mask[idx_10_30_00], "10:30:00 should be allowed");

        // 10:30:30 は許可される
        let idx_10_30_30 = 10 * 3600 + 30 * 60 + 30;
        assert!(mask[idx_10_30_30], "10:30:30 should be allowed");

        // 10:30:31 は許可されない（秒が範囲外）
        let idx_10_30_31 = 10 * 3600 + 30 * 60 + 31;
        assert!(!mask[idx_10_30_31], "10:30:31 should NOT be allowed");

        // 10:29:00 は許可されない（分が範囲外）
        let idx_10_29_00 = 10 * 3600 + 29 * 60 + 0;
        assert!(!mask[idx_10_29_00], "10:29:00 should NOT be allowed");

        // 11:00:00 は許可されない（分が範囲外：0分は30-45の範囲外）
        let idx_11_00_00 = 11 * 3600 + 0 * 60 + 0;
        assert!(!mask[idx_11_00_00], "11:00:00 should NOT be allowed (minute 0 is outside 30-45)");

        // 11:30:00 は許可される
        let idx_11_30_00 = 11 * 3600 + 30 * 60 + 0;
        assert!(mask[idx_11_30_00], "11:30:00 should be allowed");

        // 11:45:30 は許可される
        let idx_11_45_30 = 11 * 3600 + 45 * 60 + 30;
        assert!(mask[idx_11_45_30], "11:45:30 should be allowed");

        // 許可される秒数を計算
        // hour: 2時間 (10, 11)
        // minute: 16分 (30-45)
        // second: 31秒 (0-30)
        // => 2 * 16 * 31 = 992秒
        let allowed_count: usize = mask.iter().filter(|&&b| b).count();
        assert_eq!(allowed_count, 2 * 16 * 31, "Expected 992 allowed seconds");
    }

    /// 全範囲許可のテスト
    #[test]
    fn test_allowed_second_mask_full_range() {
        let range = DailyTimeRangeConfig::new(0, 23, 0, 59, 0, 59).unwrap();
        let mask = build_allowed_second_mask(&range);

        // 全86400秒が許可される
        let allowed_count: usize = mask.iter().filter(|&&b| b).count();
        assert_eq!(allowed_count, 86400, "All seconds should be allowed");
    }

    /// 日時復元のテスト（generate_display_datetime相当のロジック）
    #[test]
    fn test_datetime_restoration_from_seconds_since_2000() {
        use chrono::{DateTime, Datelike, Timelike};
        
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
        use chrono::{DateTime, Datelike, Timelike};
        
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
        use chrono::{DateTime, Datelike};
        
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

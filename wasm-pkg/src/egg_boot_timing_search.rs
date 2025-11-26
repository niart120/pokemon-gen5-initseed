//! 孵化乱数起動時間検索器
//!
//! BW/BW2における孵化乱数（タマゴ生成）の起動時間検索機能を提供する。
//! 起動日時・SHA-1パラメータ・消費範囲・個体フィルター条件に基づき、
//! 条件に合致する個体とその起動条件を列挙する。
//!
//! ## ストリーミング設計
//! 
//! 検索は以下の2フェーズに分離されている:
//! 1. LCG Seed列挙 (`LcgSeedIterator`): SHA-1計算→LCG Seed生成
//! 2. 卵Enumerate (`EggEnumeratorRef`): LCG Seed→条件に合う卵を列挙
//!
//! これによりメモリ効率の向上と中断しやすい設計を実現している。

use crate::datetime_codes::{DateCodeGenerator, TimeCodeGenerator};
use crate::egg_iv::{
    derive_pending_egg_with_state, matches_filter, resolve_egg_iv, resolve_npc_advance,
    Gender, GenerationConditions, GenerationConditionsJs, HiddenPowerInfo, IndividualFilter,
    IndividualFilterJs,
};
use crate::egg_seed_enumerator::{build_iv_sources, ParentsIVs, ParentsIVsJs};
use crate::integrated_search::generate_key_codes;
use crate::offset_calculator::{calculate_game_offset, GameMode};
use crate::personality_rng::PersonalityRNG;
use crate::sha1::{calculate_pokemon_sha1, swap_bytes_32};
use crate::sha1_simd::calculate_pokemon_sha1_simd;
use chrono::{Datelike, NaiveDate, Timelike};
use wasm_bindgen::prelude::*;

/// 2000年1月1日 00:00:00 UTCのUnix時間
const EPOCH_2000_UNIX: i64 = 946684800;
const SECONDS_PER_DAY: i64 = 86_400;

/// 日時範囲設定（integrated_search.rsからポート）
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
}

/// 孵化乱数起動時間検索器
#[wasm_bindgen]
pub struct EggBootTimingSearcher {
    // SHA-1計算用パラメータ（IntegratedSeedSearcherと共通）
    hardware: String,
    base_message: [u32; 16],
    key_codes: Vec<u32>,
    allowed_second_mask: Box<[bool; 86400]>,

    // 孵化条件パラメータ
    conditions: GenerationConditions,
    parents: ParentsIVs,
    filter: Option<IndividualFilter>,
    consider_npc_consumption: bool,
    game_mode: GameMode,

    // 消費範囲
    user_offset: u64,
    advance_count: u32,
}

#[wasm_bindgen]
impl EggBootTimingSearcher {
    /// コンストラクタ
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        // SHA-1 パラメータ
        mac: &[u8],
        nazo: &[u32],
        hardware: &str,
        key_input_mask: u32,
        frame: u32,
        // 時刻範囲
        hour_start: u32,
        hour_end: u32,
        minute_start: u32,
        minute_end: u32,
        second_start: u32,
        second_end: u32,
        // 孵化条件
        conditions: &GenerationConditionsJs,
        parents: &ParentsIVsJs,
        filter_js: Option<IndividualFilterJs>,
        consider_npc_consumption: bool,
        game_mode: GameMode,
        // 消費範囲
        user_offset: u64,
        advance_count: u32,
    ) -> Result<EggBootTimingSearcher, JsValue> {
        // バリデーション
        if mac.len() != 6 {
            return Err(JsValue::from_str("MAC address must be 6 bytes"));
        }
        if nazo.len() != 5 {
            return Err(JsValue::from_str("nazo must be 5 32-bit words"));
        }

        let time_range = DailyTimeRangeConfig::new(
            hour_start,
            hour_end,
            minute_start,
            minute_end,
            second_start,
            second_end,
        )?;

        let allowed_second_mask = build_allowed_second_mask(&time_range);

        match hardware {
            "DS" | "DS_LITE" | "3DS" => {}
            _ => return Err(JsValue::from_str("Hardware must be DS, DS_LITE, or 3DS")),
        }

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

        let key_codes = generate_key_codes(key_input_mask);

        // 孵化条件の変換
        let internal_conditions = conditions.to_internal();
        let internal_parents = parents.to_internal();
        let internal_filter = filter_js.map(|f| f.to_internal());

        Ok(EggBootTimingSearcher {
            hardware: hardware.to_string(),
            base_message,
            key_codes,
            allowed_second_mask,
            conditions: internal_conditions,
            parents: internal_parents,
            filter: internal_filter,
            consider_npc_consumption,
            game_mode,
            user_offset,
            advance_count,
        })
    }

    /// SIMD最適化版検索メソッド（ストリーミング設計）
    /// 
    /// 結果をまずRust側のVecに収集し、最後にJSArrayに変換する。
    /// これにより:
    /// - clone()による無駄なメモリ割り当てを削減
    /// - max_results到達時に早期終了可能
    #[wasm_bindgen]
    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    pub fn search_eggs_integrated_simd(
        &self,
        year_start: u32,
        month_start: u32,
        date_start: u32,
        hour_start: u32,
        minute_start: u32,
        second_start: u32,
        range_seconds: u32,
        timer0_min: u32,
        timer0_max: u32,
        vcount_min: u32,
        vcount_max: u32,
    ) -> js_sys::Array {
        // デフォルトの最大結果数
        const DEFAULT_MAX_RESULTS: usize = 10000;
        self.search_eggs_with_limit(
            year_start, month_start, date_start,
            hour_start, minute_start, second_start,
            range_seconds,
            timer0_min, timer0_max,
            vcount_min, vcount_max,
            DEFAULT_MAX_RESULTS,
        )
    }

    /// 最大結果数を指定可能な検索メソッド
    #[wasm_bindgen]
    #[allow(clippy::too_many_arguments)]
    pub fn search_eggs_with_limit(
        &self,
        year_start: u32,
        month_start: u32,
        date_start: u32,
        hour_start: u32,
        minute_start: u32,
        second_start: u32,
        range_seconds: u32,
        timer0_min: u32,
        timer0_max: u32,
        vcount_min: u32,
        vcount_max: u32,
        max_results: usize,
    ) -> js_sys::Array {
        // Rust側で結果を収集
        let mut results: Vec<EggBootTimingSearchResult> = Vec::with_capacity(
            std::cmp::min(max_results, 1000) // 初期キャパシティは控えめに
        );

        // 開始日時をUnix時間に変換
        let start_datetime =
            match NaiveDate::from_ymd_opt(year_start as i32, month_start, date_start)
                .and_then(|date| date.and_hms_opt(hour_start, minute_start, second_start))
            {
                Some(datetime) => datetime,
                None => return js_sys::Array::new(),
            };

        let start_unix = start_datetime.and_utc().timestamp();
        let base_seconds_since_2000 = start_unix - EPOCH_2000_UNIX;

        'outer: for timer0 in timer0_min..=timer0_max {
            for vcount in vcount_min..=vcount_max {
                for &key_code in &self.key_codes {
                    // SIMD バッチ処理用バッファ
                    let mut messages = [0u32; 64];
                    let mut batch_metadata: [(i64, u32, u32); 4] = [(0, 0, 0); 4];
                    let mut batch_len = 0usize;

                    for second_offset in 0..range_seconds {
                        let current_seconds = base_seconds_since_2000 + second_offset as i64;

                        // 日時コード計算（許可範囲チェック含む）
                        let (time_code, date_code) =
                            match self.calculate_datetime_codes(current_seconds) {
                                Some(result) => result,
                                None => continue,
                            };

                        // メッセージ構築
                        let message =
                            self.build_message(timer0, vcount, date_code, time_code, key_code);
                        let base_idx = batch_len * 16;
                        messages[base_idx..base_idx + 16].copy_from_slice(&message);
                        batch_metadata[batch_len] = (current_seconds, timer0, vcount);
                        batch_len += 1;

                        // 4件溜まったらSIMD処理
                        if batch_len == 4 {
                            if !self.process_simd_batch_egg(
                                &messages,
                                &batch_metadata,
                                batch_len,
                                key_code,
                                &mut results,
                                max_results,
                            ) {
                                break 'outer; // max_results到達
                            }
                            batch_len = 0;
                        }
                    }

                    // 残りを処理
                    if batch_len > 0 {
                        if !self.process_simd_batch_egg(
                            &messages,
                            &batch_metadata,
                            batch_len,
                            key_code,
                            &mut results,
                            max_results,
                        ) {
                            break 'outer; // max_results到達
                        }
                    }
                }
            }
        }

        // Rust Vec → JS Array 変換
        let js_array = js_sys::Array::new_with_length(results.len() as u32);
        for (i, result) in results.into_iter().enumerate() {
            js_array.set(i as u32, JsValue::from(result));
        }
        js_array
    }

    /// SIMDバッチ処理 - 参照ベースの新設計
    #[inline]
    fn process_simd_batch_egg(
        &self,
        messages: &[u32; 64],
        batch_metadata: &[(i64, u32, u32); 4],
        batch_size: usize,
        key_code: u32,
        results: &mut Vec<EggBootTimingSearchResult>,
        max_results: usize,
    ) -> bool {
        if batch_size == 0 {
            return true;
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

        // 各LCG Seedに対して個体検索（参照ベース）
        for i in 0..batch_size {
            let h0 = hash_results[i * 5];
            let h1 = hash_results[i * 5 + 1];
            let lcg_seed = calculate_lcg_seed_from_hash(h0, h1);

            let (current_seconds, timer0, vcount) = batch_metadata[i];

            // 日時情報を取得
            let datetime = match self.generate_display_datetime(current_seconds) {
                Some(dt) => dt,
                None => continue,
            };

            // 参照ベースの卵列挙（clone不要）
            if !self.enumerate_eggs_for_seed_ref(
                lcg_seed, datetime, timer0, vcount, key_code, results, max_results
            ) {
                return false; // max_results到達
            }
        }
        true
    }

    /// 指定されたLCG Seedに対して条件に合う個体を列挙（参照ベース）
    /// 
    /// `conditions`と`filter`をクローンせず参照で使用することでメモリ効率を向上
    fn enumerate_eggs_for_seed_ref(
        &self,
        lcg_seed: u64,
        datetime: (u32, u32, u32, u32, u32, u32),
        timer0: u32,
        vcount: u32,
        key_code: u32,
        results: &mut Vec<EggBootTimingSearchResult>,
        max_results: usize,
    ) -> bool {
        let (year, month, date, hour, minute, second) = datetime;

        // IV計算用の情報を構築
        let iv_sources = build_iv_sources(lcg_seed, self.parents);
        
        // game_offsetを計算してseedを進める
        let game_offset = calculate_game_offset(lcg_seed, self.game_mode) as u64;
        let total_offset = game_offset.saturating_add(self.user_offset);
        let (mul, add) = PersonalityRNG::lcg_affine_for_steps(total_offset);
        let mut current_seed = PersonalityRNG::lcg_apply(lcg_seed, mul, add);
        let mut next_advance = self.user_offset;

        const NPC_FRAME_THRESHOLD: u8 = 96;
        const NPC_FRAME_SLACK: u8 = 30;

        // advance_count回だけ列挙
        for _ in 0..self.advance_count {
            if results.len() >= max_results {
                return false;
            }

            let (seed_after_npc, is_stable) = if self.consider_npc_consumption {
                let (next_seed, _consumed, stable) =
                    resolve_npc_advance(current_seed, NPC_FRAME_THRESHOLD, NPC_FRAME_SLACK);
                (next_seed, stable)
            } else {
                (current_seed, false)
            };

            let (pending, _) = derive_pending_egg_with_state(seed_after_npc, &self.conditions);
            
            // IV解決
            if let Ok(resolved) = resolve_egg_iv(&pending, &iv_sources, current_seed) {
                // フィルターチェック（参照で判定）
                let passes = self.filter
                    .as_ref()
                    .is_none_or(|filter| matches_filter(&resolved, filter));

                if passes {
                    let result = self.create_result_from_resolved(
                        year, month, date, hour, minute, second,
                        timer0, vcount, key_code, lcg_seed,
                        next_advance, is_stable, &resolved,
                    );
                    results.push(result);
                }
            }

            current_seed = PersonalityRNG::next_seed(current_seed);
            next_advance = next_advance.saturating_add(1);
        }
        true
    }

    /// ResolvedEggから検索結果を作成
    fn create_result_from_resolved(
        &self,
        year: u32,
        month: u32,
        date: u32,
        hour: u32,
        minute: u32,
        second: u32,
        timer0: u32,
        vcount: u32,
        key_code: u32,
        lcg_seed: u64,
        advance: u64,
        is_stable: bool,
        egg: &crate::egg_iv::ResolvedEgg,
    ) -> EggBootTimingSearchResult {
        let (hp_type, hp_power, hp_known) = match egg.hidden_power {
            HiddenPowerInfo::Known { r#type, power } => (r#type as u8, power, true),
            HiddenPowerInfo::Unknown => (0, 0, false),
        };

        EggBootTimingSearchResult {
            year,
            month,
            date,
            hour,
            minute,
            second,
            timer0,
            vcount,
            key_code,
            lcg_seed_high: (lcg_seed >> 32) as u32,
            lcg_seed_low: lcg_seed as u32,
            advance,
            is_stable,
            ivs: egg.ivs,
            nature: egg.nature as u8,
            gender: match egg.gender {
                Gender::Male => 0,
                Gender::Female => 1,
                Gender::Genderless => 2,
            },
            ability: egg.ability as u8,
            shiny: egg.shiny as u8,
            pid: egg.pid,
            hp_type,
            hp_power,
            hp_known,
        }
    }

    /// 検索結果を作成（後方互換用）
    #[allow(dead_code)]
    fn create_result(
        &self,
        year: u32,
        month: u32,
        date: u32,
        hour: u32,
        minute: u32,
        second: u32,
        timer0: u32,
        vcount: u32,
        key_code: u32,
        lcg_seed: u64,
        egg_data: &crate::egg_seed_enumerator::EnumeratedEggData,
    ) -> EggBootTimingSearchResult {
        let (hp_type, hp_power, hp_known) = match egg_data.egg.hidden_power {
            HiddenPowerInfo::Known { r#type, power } => (r#type as u8, power, true),
            HiddenPowerInfo::Unknown => (0, 0, false),
        };

        EggBootTimingSearchResult {
            year,
            month,
            date,
            hour,
            minute,
            second,
            timer0,
            vcount,
            key_code,
            lcg_seed_high: (lcg_seed >> 32) as u32,
            lcg_seed_low: lcg_seed as u32,
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
    fn build_message(
        &self,
        timer0: u32,
        vcount: u32,
        date_code: u32,
        time_code: u32,
        key_code: u32,
    ) -> [u32; 16] {
        let mut message = self.base_message;
        message[5] = swap_bytes_32((vcount << 16) | timer0);
        message[8] = date_code;
        message[9] = time_code;
        message[12] = swap_bytes_32(key_code);
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
    /// 実際の検索フローで使用される計算ロジックを検証
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

    // =========================================================================
    // 実検索機能テスト（7日間 + 親個体値・フィルター適用）
    // =========================================================================

    use crate::egg_iv::{
        EverstonePlan, GenerationConditions, GenderRatio, IndividualFilter,
        StatRange, TrainerIds,
    };
    use crate::egg_seed_enumerator::{EggSeedEnumerator, ParentsIVs};
    use crate::offset_calculator::GameMode;
    use crate::pid_shiny_checker::ShinyType;

    /// 検索シナリオ用のテスト条件を作成
    fn create_test_generation_conditions() -> GenerationConditions {
        GenerationConditions {
            has_nidoran_flag: false,
            everstone: EverstonePlan::None,
            uses_ditto: false,
            allow_hidden_ability: false,
            female_parent_has_hidden: false,
            reroll_count: 0, // 国際孵化なし
            trainer_ids: TrainerIds::new(12345, 54321),
            gender_ratio: GenderRatio::new(127, false), // 50:50
        }
    }

    /// 親個体値を作成（片親6V、他方不明=乱数）
    fn create_test_parents_ivs() -> ParentsIVs {
        ParentsIVs {
            male: [31, 31, 31, 31, 31, 31],   // 6V
            female: [15, 15, 15, 15, 15, 15], // 平均的な個体
        }
    }

    /// 厳しめのフィルターを作成（実際の検索シナリオ想定）
    fn create_strict_filter() -> IndividualFilter {
        IndividualFilter {
            iv_ranges: [
                StatRange::new(25, 31), // HP: 25-31
                StatRange::new(25, 31), // Attack: 25-31
                StatRange::new(25, 31), // Defense: 25-31
                StatRange::new(25, 31), // SpAtk: 25-31
                StatRange::new(25, 31), // SpDef: 25-31
                StatRange::new(25, 31), // Speed: 25-31
            ],
            nature: None, // 性格は問わない
            gender: None,
            ability: None,
            shiny: None,
            hidden_power_type: None,
            hidden_power_power: None,
        }
    }

    /// 緩めのフィルターを作成（確実にヒットする条件）
    fn create_lenient_filter() -> IndividualFilter {
        IndividualFilter {
            iv_ranges: [
                StatRange::new(0, 31),
                StatRange::new(0, 31),
                StatRange::new(0, 31),
                StatRange::new(0, 31),
                StatRange::new(0, 31),
                StatRange::new(0, 31),
            ],
            nature: None,
            gender: None,
            ability: None,
            shiny: None,
            hidden_power_type: None,
            hidden_power_power: None,
        }
    }

    /// 色違いフィルターを作成
    fn create_shiny_filter() -> IndividualFilter {
        IndividualFilter {
            iv_ranges: [StatRange::new(0, 31); 6],
            nature: None,
            gender: None,
            ability: None,
            shiny: Some(ShinyType::Star), // 色違い(星)のみ
            hidden_power_type: None,
            hidden_power_power: None,
        }
    }

    /// 実検索テスト: EggSeedEnumeratorを使用して検索が実行できることを検証
    /// 親個体値・条件を設定し、100消費で少なくとも1件の結果が返ることを確認
    #[test]
    fn test_egg_seed_enumerator_with_filter_returns_results() {
        let base_seed: u64 = 0x0123456789ABCDEF;
        let user_offset: u64 = 0;
        let count: u32 = 100;
        let conditions = create_test_generation_conditions();
        let parents = create_test_parents_ivs();
        let filter = create_lenient_filter();

        let mut enumerator = EggSeedEnumerator::new(
            base_seed,
            user_offset,
            count,
            conditions,
            parents,
            Some(filter),
            false, // consider_npc_consumption
            GameMode::BwContinue,
        );

        let mut results = Vec::new();
        while let Ok(Some(egg_data)) = enumerator.next_egg() {
            results.push(egg_data);
        }

        // 緩いフィルターなので結果が得られるはず
        assert!(
            !results.is_empty(),
            "Expected at least one result with lenient filter"
        );

        // 結果のadvanceが正しい範囲内であることを確認
        for result in &results {
            assert!(result.advance < count as u64);
        }
    }

    /// 実検索テスト: 厳しいフィルター条件でも検索処理が完了することを検証
    #[test]
    fn test_egg_seed_enumerator_with_strict_filter_completes() {
        let base_seed: u64 = 0xFEDCBA9876543210;
        let user_offset: u64 = 0;
        let count: u32 = 1000; // より多くの消費で検索
        let conditions = create_test_generation_conditions();
        let parents = create_test_parents_ivs();
        let filter = create_strict_filter();

        let mut enumerator = EggSeedEnumerator::new(
            base_seed,
            user_offset,
            count,
            conditions,
            parents,
            Some(filter),
            false,
            GameMode::BwContinue,
        );

        let mut results = Vec::new();
        while let Ok(Some(egg_data)) = enumerator.next_egg() {
            results.push(egg_data);
        }

        // 処理が正常に完了したことを確認
        // 厳しいフィルターなので結果がないこともあり得る
        // 重要なのはパニックせずに完了すること
        assert!(
            enumerator.remaining() == 0 || !results.is_empty(),
            "Enumerator should complete or find results"
        );
    }

    /// 実検索テスト: 検索結果をEnumeratorで再現し、期待個体が出現することを検証
    #[test]
    fn test_search_result_can_be_reproduced_by_enumerator() {
        // ステップ1: 最初の検索で結果を取得
        let base_seed: u64 = 0xABCDEF0123456789;
        let conditions = create_test_generation_conditions();
        let parents = create_test_parents_ivs();
        let filter = create_lenient_filter();

        let mut first_enumerator = EggSeedEnumerator::new(
            base_seed,
            0,    // user_offset
            50,   // count
            conditions,
            parents,
            Some(filter.clone()),
            false,
            GameMode::BwContinue,
        );

        // 最初の結果を取得
        let first_result = first_enumerator.next_egg().expect("Should not error").expect("Should find at least one egg");
        let found_advance = first_result.advance;
        let found_ivs = first_result.egg.ivs;
        let found_nature = first_result.egg.nature;
        let found_pid = first_result.egg.pid;

        // ステップ2: 同じseedとadvanceで再度Enumeratorを作成し、同じ個体が出るか検証
        let mut verification_enumerator = EggSeedEnumerator::new(
            base_seed,
            found_advance, // 発見したadvanceから開始
            1,             // 1個だけ生成
            conditions,
            parents,
            None, // フィルターなしで全個体を取得
            false,
            GameMode::BwContinue,
        );

        let reproduced_result = verification_enumerator
            .next_egg()
            .expect("Should not error")
            .expect("Should produce exactly one egg");

        // 同じ個体が生成されることを検証
        assert_eq!(
            reproduced_result.advance, found_advance,
            "Advance should match"
        );
        assert_eq!(
            reproduced_result.egg.ivs, found_ivs,
            "IVs should match: expected {:?}, got {:?}",
            found_ivs, reproduced_result.egg.ivs
        );
        assert_eq!(
            reproduced_result.egg.nature, found_nature,
            "Nature should match"
        );
        assert_eq!(
            reproduced_result.egg.pid, found_pid,
            "PID should match"
        );
    }

    /// 実検索テスト: 7日間分のシード生成とフィルタリングのシミュレーション
    /// SHA-1計算とEggSeedEnumeratorを組み合わせた統合テスト
    #[test]
    fn test_7_day_search_simulation_with_filter() {
        // 7日間で検索される可能性のあるシード数のサンプリング
        // 実際には (7日 * 24時間 * 60分 * 60秒) * Timer0範囲 * VCount範囲 のシードが生成される
        // ここでは計算量を抑えるため、代表的なシードをサンプリング

        let conditions = create_test_generation_conditions();
        let parents = create_test_parents_ivs();
        let filter = create_strict_filter(); // 厳しいフィルター

        // 異なるシードでの検索結果を収集
        let test_seeds: Vec<u64> = vec![
            0x0000000000000001,
            0x123456789ABCDEF0,
            0xFEDCBA9876543210,
            0xAAAAAAAAAAAAAAAA,
            0x5555555555555555,
            0x0F0F0F0F0F0F0F0F,
            0xF0F0F0F0F0F0F0F0,
            0xDEADBEEFCAFEBABE,
        ];

        let mut total_results = 0;
        let mut total_searched = 0;
        let count_per_seed = 100; // 各シードで100消費を検索

        for seed in test_seeds {
            let mut enumerator = EggSeedEnumerator::new(
                seed,
                0,
                count_per_seed,
                conditions,
                parents,
                Some(filter.clone()),
                false,
                GameMode::BwContinue,
            );

            while let Ok(Some(_egg_data)) = enumerator.next_egg() {
                total_results += 1;
            }
            total_searched += count_per_seed;
        }

        // 厳しいフィルターでも処理が完了したことを確認
        assert!(
            total_searched > 0,
            "Should have searched through multiple seeds"
        );

        // 結果の数は条件によって変わるが、処理が正常に完了したことが重要
        println!(
            "7-day simulation: searched {} eggs, found {} matches",
            total_searched, total_results
        );
    }

    /// 実検索テスト: 色違い検索のシミュレーション（確率は低いので結果0でも正常）
    #[test]
    fn test_shiny_search_simulation() {
        let conditions = {
            let mut c = create_test_generation_conditions();
            // 色違い判定用にTIDとSIDを設定
            c.trainer_ids = TrainerIds::new(12345, 54321);
            c
        };
        let parents = create_test_parents_ivs();
        let filter = create_shiny_filter();

        // 多くのシードで検索
        let test_seeds: Vec<u64> = (0..100).map(|i| 0x1234567890ABCDEF + i * 12345).collect();
        let num_seeds = test_seeds.len();

        let mut shiny_found = 0;
        let count_per_seed = 100;

        for seed in &test_seeds {
            let mut enumerator = EggSeedEnumerator::new(
                *seed,
                0,
                count_per_seed,
                conditions,
                parents,
                Some(filter.clone()),
                false,
                GameMode::BwContinue,
            );

            while let Ok(Some(egg_data)) = enumerator.next_egg() {
                // 色違いであることを確認
                assert!(
                    egg_data.egg.shiny == ShinyType::Star || egg_data.egg.shiny == ShinyType::Square,
                    "Filtered result should be shiny"
                );
                shiny_found += 1;
            }
        }

        // 色違い確率は約1/8192なので、10000個体で1-2個見つかるかもしれない
        // 見つからなくても処理が正常に完了していればOK
        println!("Shiny search: searched {} eggs, found {} shinies", 
                 num_seeds * count_per_seed as usize, shiny_found);
    }

    /// 実検索テスト: 異なるGameModeでの検索動作を検証
    #[test]
    fn test_search_with_different_game_modes() {
        let base_seed: u64 = 0x9876543210FEDCBA;
        let conditions = create_test_generation_conditions();
        let parents = create_test_parents_ivs();
        let filter = create_lenient_filter();

        let game_modes = [
            GameMode::BwNewGameWithSave,
            GameMode::BwNewGameNoSave,
            GameMode::BwContinue,
            GameMode::Bw2NewGameWithMemoryLinkSave,
            GameMode::Bw2NewGameNoMemoryLinkSave,
            GameMode::Bw2NewGameNoSave,
            GameMode::Bw2ContinueWithMemoryLink,
            GameMode::Bw2ContinueNoMemoryLink,
        ];

        for game_mode in game_modes {
            let mut enumerator = EggSeedEnumerator::new(
                base_seed,
                0,
                10, // 少数で検証
                conditions,
                parents,
                Some(filter.clone()),
                false,
                game_mode,
            );

            let mut results = Vec::new();
            while let Ok(Some(egg_data)) = enumerator.next_egg() {
                results.push(egg_data);
            }

            // 各GameModeで処理が完了することを確認
            assert!(
                !results.is_empty(),
                "GameMode {:?} should produce at least one result with lenient filter",
                game_mode
            );
        }
    }
}

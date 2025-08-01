/// 統合シード探索システム
/// メッセージ生成とSHA-1計算を一体化し、WebAssembly内で完結する高速探索を実現
use wasm_bindgen::prelude::*;
use std::collections::BTreeSet;
use crate::datetime_codes::{TimeCodeGenerator, DateCodeGenerator};

// コンパイル時最適化のためのアトリビュート
#[cfg(target_family = "wasm")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;
use crate::sha1::{calculate_pokemon_sha1, swap_bytes_32};
use chrono::{NaiveDate, Datelike, Timelike};

// Import the `console.log` function from the browser console
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

/// 2000年1月1日 00:00:00 UTCのUnix時間
const EPOCH_2000_UNIX: i64 = 946684800;

/// 検索パラメータ構造体（内部用）
#[derive(Clone, Copy)]
struct SearchParams {
    timer0: u32,
    vcount: u32,
}

/// ハッシュ値構造体（内部用）
#[derive(Clone, Copy)]
struct HashValues {
    h0: u32,
    h1: u32,
    h2: u32,
    h3: u32,
    h4: u32,
}

/// 探索結果構造体
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct SearchResult {
    seed: u32,
    hash: String,
    year: u32,
    month: u32,
    date: u32,
    hour: u32,
    minute: u32,
    second: u32,
    timer0: u32,
    vcount: u32,
}

#[wasm_bindgen]
impl SearchResult {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]  // WebAssembly constructor requires all parameters
    pub fn new(seed: u32, hash: String, year: u32, month: u32, date: u32, hour: u32, minute: u32, second: u32, timer0: u32, vcount: u32) -> SearchResult {
        SearchResult { seed, hash, year, month, date, hour, minute, second, timer0, vcount }
    }
    
    #[wasm_bindgen(getter)]
    pub fn seed(&self) -> u32 { self.seed }
    #[wasm_bindgen(getter)]
    pub fn hash(&self) -> String { self.hash.clone() }
    #[wasm_bindgen(getter)]
    pub fn year(&self) -> u32 { self.year }
    #[wasm_bindgen(getter)]
    pub fn month(&self) -> u32 { self.month }
    #[wasm_bindgen(getter)]
    pub fn date(&self) -> u32 { self.date }
    #[wasm_bindgen(getter)]
    pub fn hour(&self) -> u32 { self.hour }
    #[wasm_bindgen(getter)]
    pub fn minute(&self) -> u32 { self.minute }
    #[wasm_bindgen(getter)]
    pub fn second(&self) -> u32 { self.second }
    #[wasm_bindgen(getter)]
    pub fn timer0(&self) -> u32 { self.timer0 }
    #[wasm_bindgen(getter)]
    pub fn vcount(&self) -> u32 { self.vcount }
}

/// 統合シード探索器
/// 固定パラメータを事前計算し、日時範囲を高速探索する
#[wasm_bindgen]
pub struct IntegratedSeedSearcher {
    // 実行時に必要なパラメータ
    hardware: String,
    
    // キャッシュされた基本メッセージ
    base_message: [u32; 16],
}

#[wasm_bindgen]
impl IntegratedSeedSearcher {
    /// コンストラクタ: 固定パラメータの事前計算
    #[wasm_bindgen(constructor)]
    pub fn new(mac: &[u8], nazo: &[u32], hardware: &str, key_input: u32, frame: u32) -> Result<IntegratedSeedSearcher, JsValue> {
        // バリデーション
        if mac.len() != 6 {
            return Err(JsValue::from_str("MAC address must be 6 bytes"));
        }
        if nazo.len() != 5 {
            return Err(JsValue::from_str("nazo must be 5 32-bit words"));
        }
        
        // Hardware type validation
        match hardware {
            "DS" | "DS_LITE" | "3DS" => {},
            _ => return Err(JsValue::from_str("Hardware must be DS, DS_LITE, or 3DS")),
        }

        // 基本メッセージテンプレートを事前構築（TypeScript側レイアウトに準拠）
        let mut base_message = [0u32; 16];
        
        // data[0-4]: Nazo values (little-endian conversion already applied)
        for i in 0..5 {
            base_message[i] = swap_bytes_32(nazo[i]);
        }
        
        // data[5]: (VCount << 16) | Timer0 - 動的に設定
        // data[6]: MAC address lower 16 bits (no endian conversion)
        let mac_lower = ((mac[4] as u32) << 8) | (mac[5] as u32);
        base_message[6] = mac_lower;
        
        // data[7]: MAC address upper 32 bits XOR GxStat XOR Frame (little-endian conversion needed)
        let mac_upper = (mac[0] as u32) | ((mac[1] as u32) << 8) | ((mac[2] as u32) << 16) | ((mac[3] as u32) << 24);
        let gx_stat = 0x06000000u32;
        let data7 = mac_upper ^ gx_stat ^ frame;
        base_message[7] = swap_bytes_32(data7);
        
        // data[8]: Date (YYMMDDWW format) - 動的に設定
        // data[9]: Time (HHMMSS00 format + PM flag) - 動的に設定
        // data[10-11]: Fixed values 0x00000000
        base_message[10] = 0x00000000;
        base_message[11] = 0x00000000;
        
        // data[12]: Key input (now configurable)
        base_message[12] = swap_bytes_32(key_input);
        
        // data[13-15]: SHA-1 padding
        base_message[13] = 0x80000000;
        base_message[14] = 0x00000000;
        base_message[15] = 0x000001A0;

        Ok(IntegratedSeedSearcher {
            hardware: hardware.to_string(),
            base_message,
        })
    }

    /// 統合シード探索メイン関数
    /// 日時範囲とTimer0/VCount範囲を指定して一括探索
    #[wasm_bindgen]
    #[allow(clippy::too_many_arguments)]  // Search function requires comprehensive parameters
    #[inline(never)]  // 大きな関数はinlineしない
    pub fn search_seeds_integrated(
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
        target_seeds: &[u32],
    ) -> js_sys::Array {
        
        let results = js_sys::Array::new();

        // Target seedsをBTreeSetに変換して最適化されたルックアップを実現
        let target_set: BTreeSet<u32> = target_seeds.iter().cloned().collect();

        // 開始日時をUnix時間に変換（ループ外で1回のみ実行）
        let start_datetime = match NaiveDate::from_ymd_opt(year_start as i32, month_start, date_start)
            .and_then(|date| date.and_hms_opt(hour_start, minute_start, second_start)) 
        {
            Some(datetime) => datetime,
            None => {
                return results;
            }
        };
        
        let start_unix = start_datetime.and_utc().timestamp();
        let base_seconds_since_2000 = start_unix - EPOCH_2000_UNIX;

        // 外側ループ: Timer0とVCount（SIMD版と同様の構造）
        for timer0 in timer0_min..=timer0_max {
            for vcount in vcount_min..=vcount_max {
                
                // 内側ループ: 日時範囲の探索
                for second_offset in 0..range_seconds {
                    let current_seconds_since_2000 = base_seconds_since_2000 + second_offset as i64;
                    
                    let (time_code, date_code) = match self.calculate_datetime_codes(current_seconds_since_2000) {
                        Some(result) => result,
                        None => continue,
                    };
                    
                    // メッセージを構築してSHA-1計算
                    let message = self.build_message(timer0, vcount, date_code, time_code);
                    let (h0, h1, h2, h3, h4) = calculate_pokemon_sha1(&message);
                    let seed = crate::sha1::calculate_pokemon_seed_from_hash(h0, h1);
                    
                    // ターゲットシードマッチ時のみ日時とハッシュを生成
                    let hash_values = HashValues { h0, h1, h2, h3, h4 };
                    let params = SearchParams {
                        timer0,
                        vcount,
                    };
                    self.check_and_add_result(seed, &hash_values, current_seconds_since_2000, &params, &target_set, &results);
                }
            }
        }

        results
    }

    /// 統合シード探索SIMD版
    /// range_secondsを最内ループに配置してSIMD SHA-1計算を活用
    #[wasm_bindgen]
    #[allow(clippy::too_many_arguments)]  // SIMD search function requires comprehensive parameters
    #[inline(never)]  // 大きな関数はinlineしない
    pub fn search_seeds_integrated_simd(
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
        target_seeds: &[u32],
    ) -> js_sys::Array {
        
        let results = js_sys::Array::new();

        // Target seedsをBTreeSetに変換して最適化されたルックアップを実現
        let target_set: BTreeSet<u32> = target_seeds.iter().cloned().collect();

        // 開始日時をUnix時間に変換（ループ外で1回のみ実行）
        let start_datetime = match NaiveDate::from_ymd_opt(year_start as i32, month_start, date_start)
            .and_then(|date| date.and_hms_opt(hour_start, minute_start, second_start)) 
        {
            Some(datetime) => datetime,
            None => {
                return results;
            }
        };
        
        let start_unix = start_datetime.and_utc().timestamp();
        let base_seconds_since_2000 = start_unix - EPOCH_2000_UNIX;

        // 外側ループ: Timer0とVCount
        for timer0 in timer0_min..=timer0_max {
            for vcount in vcount_min..=vcount_max {
                for second_offset in (0..range_seconds).step_by(4) {
                    let batch_size = std::cmp::min(4, range_seconds - second_offset);
                    
                    if batch_size == 4 {
                        // 4つの秒を並列処理
                        let params = SearchParams {
                            timer0,
                            vcount,
                        };
                        self.process_simd_batch(
                            second_offset, 
                            base_seconds_since_2000, 
                            &params,
                            &target_set, 
                            &results
                        );
                    } else {
                        // 残りの秒を個別処理
                        let params = SearchParams {
                            timer0,
                            vcount,
                        };
                        self.process_remaining_seconds(
                            second_offset, 
                            batch_size, 
                            base_seconds_since_2000, 
                            &params,
                            &target_set, 
                            &results
                        );
                    }
                }
            }
        }

        results
    }

    /// SIMD バッチ処理（4つの秒を並列計算）
    #[inline]
    fn process_simd_batch(
        &self,
        second_offset: u32,
        base_seconds_since_2000: i64,
        params: &SearchParams,
        target_seeds: &BTreeSet<u32>,
        results: &js_sys::Array,
    ) {
        let mut messages = [0u32; 64]; // 4組 × 16ワード
        let mut valid_messages = [true; 4];
        let mut seconds_batch = [0i64; 4]; // 結果生成用
        
        // Timer0/VCountの値を事前に計算（SIMD用）
        let timer_vcount_value = (params.vcount << 16) | params.timer0;
        let swapped_timer_vcount = crate::sha1::swap_bytes_32(timer_vcount_value);
        
        // 4つのメッセージを準備
        for i in 0..4 {
            let current_second_offset = second_offset + i as u32;
            let current_seconds_since_2000 = base_seconds_since_2000 + current_second_offset as i64;
            
            let (time_code, date_code) = match self.calculate_datetime_codes(current_seconds_since_2000) {
                Some(result) => result,
                None => {
                    valid_messages[i] = false;
                    continue;
                }
            };
            
            seconds_batch[i] = current_seconds_since_2000;
            
            // メッセージを構築してバッチに追加（バイトスワップ済み値を使用）
            let mut message = self.base_message;
            message[5] = swapped_timer_vcount; // 事前計算済み
            message[8] = date_code;
            message[9] = time_code;
            
            let base_idx = i * 16;
            messages[base_idx..base_idx + 16].copy_from_slice(&message);
        }
        
        // SIMD SHA-1計算を実行
        let hash_results = crate::sha1_simd::calculate_pokemon_sha1_simd(&messages);
        
        // 各組の結果を処理
        for i in 0..4 {
            if !valid_messages[i] {
                continue;
            }
            
            let h0 = hash_results[i * 5];
            let h1 = hash_results[i * 5 + 1];
            let h2 = hash_results[i * 5 + 2];
            let h3 = hash_results[i * 5 + 3];
            let h4 = hash_results[i * 5 + 4];
            let seed = crate::sha1::calculate_pokemon_seed_from_hash(h0, h1);
            
            // マッチ時のみ日時とハッシュを生成
            let hash_values = HashValues { h0, h1, h2, h3, h4 };
            let params_for_result = SearchParams {
                timer0: params.timer0,
                vcount: params.vcount,
            };
            self.check_and_add_result(seed, &hash_values, seconds_batch[i], &params_for_result, target_seeds, results);
        }
    }

    /// 端数秒の個別処理（非SIMD）
    #[inline]
    fn process_remaining_seconds(
        &self,
        second_offset: u32,
        batch_size: u32,
        base_seconds_since_2000: i64,
        params: &SearchParams,
        target_seeds: &BTreeSet<u32>,
        results: &js_sys::Array,
    ) {
        for i in 0..batch_size {
            let current_second_offset = second_offset + i;
            let current_seconds_since_2000 = base_seconds_since_2000 + current_second_offset as i64;
            
            let (time_code, date_code) = match self.calculate_datetime_codes(current_seconds_since_2000) {
                Some(result) => result,
                None => continue,
            };
            
            // メッセージを構築してSHA-1計算
            let message = self.build_message(params.timer0, params.vcount, date_code, time_code);
            let (h0, h1, h2, h3, h4) = crate::sha1::calculate_pokemon_sha1(&message);
            let seed = crate::sha1::calculate_pokemon_seed_from_hash(h0, h1);
            
            // マッチ時のみ日時とハッシュを生成
            let hash_values = HashValues { h0, h1, h2, h3, h4 };
            self.check_and_add_result(seed, &hash_values, current_seconds_since_2000, params, target_seeds, results);
        }
    }

    /// 日時コード生成（結果表示用日時は遅延生成）
    #[inline(always)]
    fn calculate_datetime_codes(&self, seconds_since_2000: i64) -> Option<(u32, u32)> {
        if seconds_since_2000 < 0 {
            return None;
        }
        
        let time_index = (seconds_since_2000 % 86400) as u32;
        let date_index = (seconds_since_2000 / 86400) as u32;

        let time_code = TimeCodeGenerator::get_time_code_for_hardware(time_index, &self.hardware);
        let date_code = DateCodeGenerator::get_date_code(date_index);
        
        Some((time_code, date_code))
    }

    /// 結果表示用の日時を生成（マッチした場合のみ）
    fn generate_display_datetime(&self, seconds_since_2000: i64) -> Option<(u32, u32, u32, u32, u32, u32)> {
        let result_datetime = chrono::DateTime::from_timestamp(seconds_since_2000 + EPOCH_2000_UNIX, 0)?
            .naive_utc();
        
        Some((
            result_datetime.year() as u32,
            result_datetime.month(),
            result_datetime.day(),
            result_datetime.hour(),
            result_datetime.minute(),
            result_datetime.second()
        ))
    }

    /// メッセージ構築の共通処理
    #[inline(always)]
    fn build_message(&self, timer0: u32, vcount: u32, date_code: u32, time_code: u32) -> [u32; 16] {
        let mut message = self.base_message;
        message[5] = crate::sha1::swap_bytes_32((vcount << 16) | timer0);
        message[8] = date_code;
        message[9] = time_code;
        message
    }

    /// ハッシュ値を16進数文字列に変換
    #[inline]
    fn hash_to_hex_string(&self, h0: u32, h1: u32, h2: u32, h3: u32, h4: u32) -> String {
        format!("{h0:08x}{h1:08x}{h2:08x}{h3:08x}{h4:08x}")
    }

    /// 結果チェックと追加（マッチ時のみ日時とハッシュを生成）
    #[inline(always)]
    fn check_and_add_result(
        &self,
        seed: u32,
        hash_values: &HashValues,
        seconds_since_2000: i64,
        params: &SearchParams,
        target_seeds: &BTreeSet<u32>,
        results: &js_sys::Array,
    ) {
        if target_seeds.contains(&seed) {
            // マッチした場合のみ日時とハッシュを生成
            if let Some(datetime) = self.generate_display_datetime(seconds_since_2000) {
                let (year, month, date, hour, minute, second) = datetime;
                let hash = self.hash_to_hex_string(hash_values.h0, hash_values.h1, hash_values.h2, hash_values.h3, hash_values.h4);
                let result = SearchResult::new(seed, hash, year, month, date, hour, minute, second, params.timer0, params.vcount);
                results.push(&JsValue::from(result));
            }
        }
    }
}

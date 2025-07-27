/// 統合シード探索システム
/// メッセージ生成とSHA-1計算を一体化し、WebAssembly内で完結する高速探索を実現
use wasm_bindgen::prelude::*;
use crate::datetime_codes::{TimeCodeGenerator, DateCodeGenerator};
use crate::sha1::{calculate_pokemon_sha1, to_little_endian_32};

// Import the `console.log` function from the browser console
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

// Define a macro to make console.log easier to use
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

/// 探索結果構造体
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct SearchResult {
    seed: u32,
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
    pub fn new(seed: u32, year: u32, month: u32, date: u32, hour: u32, minute: u32, second: u32, timer0: u32, vcount: u32) -> SearchResult {
        SearchResult { seed, year, month, date, hour, minute, second, timer0, vcount }
    }
    
    #[wasm_bindgen(getter)]
    pub fn seed(&self) -> u32 { self.seed }
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
    // 事前計算された固定パラメータ（将来の拡張用に保持）
    #[allow(dead_code)]
    mac_le: [u32; 2],
    #[allow(dead_code)]
    nazo: [u32; 5],
    #[allow(dead_code)]
    version: u32,
    #[allow(dead_code)]
    frame: u32,
    #[allow(dead_code)]
    hardware: String,
    #[allow(dead_code)]
    key_input: u32,
    
    // キャッシュされた基本メッセージ
    base_message: [u32; 16],
}

#[wasm_bindgen]
impl IntegratedSeedSearcher {
    /// コンストラクタ: 固定パラメータの事前計算
    #[wasm_bindgen(constructor)]
    pub fn new(mac: &[u8], nazo: &[u32], hardware: &str, key_input: u32, _version: u32, frame: u32) -> Result<IntegratedSeedSearcher, JsValue> {
        // 初期化時のみログ出力（メモリリーク対策：1回のみ実行）
        console_log!("🔧 IntegratedSeedSearcher initialized - Hardware: {}, KeyInput: 0x{:X}, Tables: Time={}, Date={}", 
            hardware, key_input, TimeCodeGenerator::TIME_CODES.len(), DateCodeGenerator::DATE_CODES.len());
        
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

        // MACアドレス配列をそのまま保持（直接使用）
        if mac.len() != 6 {
            return Err(JsValue::from_str("MAC address must be 6 bytes"));
        }
        if nazo.len() != 5 {
            return Err(JsValue::from_str("nazo must be 5 32-bit words"));
        }

        // nazoをコピー
        let mut nazo_array = [0u32; 5];
        nazo_array.copy_from_slice(nazo);

        // 基本メッセージテンプレートを事前構築（TypeScript側レイアウトに準拠）
        let mut base_message = [0u32; 16];
        
        // data[0-4]: Nazo values (little-endian conversion already applied)
        for i in 0..5 {
            base_message[i] = to_little_endian_32(nazo_array[i]);
        }
        
        // data[5]: (VCount << 16) | Timer0 - 動的に設定
        // data[6]: MAC address lower 16 bits (no endian conversion)
        let mac_lower = ((mac[4] as u32) << 8) | (mac[5] as u32);
        base_message[6] = mac_lower;
        
        // data[7]: MAC address upper 32 bits XOR GxStat XOR Frame (little-endian conversion needed)
        let mac_upper = (mac[0] as u32) | ((mac[1] as u32) << 8) | ((mac[2] as u32) << 16) | ((mac[3] as u32) << 24);
        let gx_stat = 0x06000000u32;
        let data7 = mac_upper ^ gx_stat ^ frame;
        base_message[7] = to_little_endian_32(data7);
        
        // data[8]: Date (YYMMDDWW format) - 動的に設定
        // data[9]: Time (HHMMSS00 format + PM flag) - 動的に設定
        // data[10-11]: Fixed values 0x00000000
        base_message[10] = 0x00000000;
        base_message[11] = 0x00000000;
        
        // data[12]: Key input (now configurable)
        base_message[12] = to_little_endian_32(key_input);
        
        // data[13-15]: SHA-1 padding
        base_message[13] = 0x80000000;
        base_message[14] = 0x00000000;
        base_message[15] = 0x000001A0;

        Ok(IntegratedSeedSearcher {
            mac_le: [0, 0], // 使用しないため仮値
            nazo: nazo_array,
            version: _version,
            frame,
            hardware: hardware.to_string(),
            key_input,
            base_message,
        })
    }

    /// 統合シード探索メイン関数
    /// 日時範囲とTimer0/VCount範囲を指定して一括探索
    #[wasm_bindgen]
    #[allow(clippy::too_many_arguments)]  // Search function requires comprehensive parameters
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

        // 日時範囲の探索
        for second_offset in 0..range_seconds {
            // 現在の秒数計算
            let mut current_second = second_start + second_offset;
            let mut current_minute = minute_start;
            let mut current_hour = hour_start;
            let mut current_date = date_start;
            let current_month = month_start;
            let current_year = year_start;

            // 時刻の正規化
            if current_second >= 60 {
                current_minute += current_second / 60;
                current_second %= 60;
            }
            if current_minute >= 60 {
                current_hour += current_minute / 60;
                current_minute %= 60;
            }
            if current_hour >= 24 {
                current_date += current_hour / 24;
                current_hour %= 24;
            }

            // 事前計算テーブルから日時コードを高速取得（hardware-specific）
            let time_code = TimeCodeGenerator::get_time_code_for_hardware(current_hour, current_minute, current_second, &self.hardware);
            let date_code = DateCodeGenerator::get_date_code(current_year, current_month, current_date);

            // Timer0とVCountの範囲探索
            for timer0 in timer0_min..=timer0_max {
                for vcount in vcount_min..=vcount_max {
                    // メッセージを動的に構築（TypeScript側レイアウトに準拠）
                    let mut message = self.base_message;
                    
                    // data[5]: (VCount << 16) | Timer0 (little-endian conversion needed)
                    message[5] = to_little_endian_32((vcount << 16) | timer0);
                    
                    // data[8]: Date (YYMMDDWW format) - no endian conversion
                    message[8] = date_code;
                    
                    // data[9]: Time (HHMMSS00 format + PM flag) - no endian conversion  
                    message[9] = time_code;

                    // SHA-1計算
                    let (h0, _h1) = calculate_pokemon_sha1(&message);
                    let seed = h0;

                    // ターゲットシードと照合
                    for &target in target_seeds {
                        if seed == target {
                            let result = SearchResult::new(
                                seed,
                                current_year,
                                current_month,
                                current_date,
                                current_hour,
                                current_minute,
                                current_second,
                                timer0,
                                vcount,
                            );
                            results.push(&JsValue::from(result));
                        }
                    }
                }
            }
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "wasm32")]  // WebAssembly環境でのみテスト実行
    fn test_integrated_searcher_creation() {
        let mac = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC];
        let nazo = [0x02215f10, 0x01000000, 0xc0000000, 0x00007fff, 0x00000000];
        
        let searcher = IntegratedSeedSearcher::new(&mac, &nazo, 5, 8);
        assert!(searcher.is_ok());
    }
    
    #[test]
    fn test_search_result() {
        let result = SearchResult::new(0x12345678, 2012, 6, 15, 10, 30, 45, 1120, 50);
        assert_eq!(result.seed(), 0x12345678);
        assert_eq!(result.year(), 2012);
        assert_eq!(result.month(), 6);
    }
}

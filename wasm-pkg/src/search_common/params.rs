//! パラメータ型定義モジュール
//!
//! 検索パラメータの内部型と公開型を定義する。

use super::{EPOCH_2000_UNIX, HARDWARE_FRAME_3DS, HARDWARE_FRAME_DS, HARDWARE_FRAME_DS_LITE};
use chrono::NaiveDate;
use wasm_bindgen::prelude::*;

// =============================================================================
// ローカルヘルパー（循環参照回避用）
// =============================================================================

/// 日付をseconds_since_2000に変換（循環参照回避用ローカル実装）
fn date_to_seconds_since_2000_local(year: u32, month: u32, day: u32) -> Option<i64> {
    let datetime = NaiveDate::from_ymd_opt(year as i32, month, day)?.and_hms_opt(0, 0, 0)?;
    let unix = datetime.and_utc().timestamp();
    Some(unix - EPOCH_2000_UNIX)
}

// =============================================================================
// Hardware関連
// =============================================================================

/// Hardwareタイプ
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareType {
    DS,
    DSLite,
    ThreeDS,
}

impl HardwareType {
    /// 文字列からHardwareTypeを生成
    pub fn from_str(hardware: &str) -> Result<Self, &'static str> {
        match hardware {
            "DS" => Ok(HardwareType::DS),
            "DS_LITE" => Ok(HardwareType::DSLite),
            "3DS" => Ok(HardwareType::ThreeDS),
            _ => Err("Hardware must be DS, DS_LITE, or 3DS"),
        }
    }

    /// frame値を取得
    pub fn frame(&self) -> u32 {
        match self {
            HardwareType::DS => HARDWARE_FRAME_DS,
            HardwareType::DSLite => HARDWARE_FRAME_DS_LITE,
            HardwareType::ThreeDS => HARDWARE_FRAME_3DS,
        }
    }

    /// 文字列表現を取得（TimeCodeGenerator用）
    pub fn as_str(&self) -> &'static str {
        match self {
            HardwareType::DS => "DS",
            HardwareType::DSLite => "DS_LITE",
            HardwareType::ThreeDS => "3DS",
        }
    }
}

// =============================================================================
// 内部型: パラメータ
// =============================================================================

/// DS設定パラメータ（内部型）
#[derive(Debug, Clone)]
pub struct DSConfig {
    pub mac: [u8; 6],
    pub nazo: [u32; 5],
    pub hardware: HardwareType,
}

impl DSConfig {
    pub fn new(mac: [u8; 6], nazo: [u32; 5], hardware: HardwareType) -> Self {
        Self { mac, nazo, hardware }
    }

    /// frame値を取得
    pub fn frame(&self) -> u32 {
        self.hardware.frame()
    }
}

/// セグメントパラメータ（内部型）
#[derive(Debug, Clone, Copy)]
pub struct SegmentParams {
    pub timer0: u32,
    pub vcount: u32,
    pub key_code: u32,
}

impl SegmentParams {
    pub fn new(timer0: u32, vcount: u32, key_code: u32) -> Self {
        Self {
            timer0,
            vcount,
            key_code,
        }
    }
}

/// 時刻範囲パラメータ（内部型）
///
/// 1日の中で検索対象とする時刻範囲を指定する。
#[derive(Debug, Clone, Copy)]
pub struct TimeRangeParams {
    pub hour_start: u32,
    pub hour_end: u32,
    pub minute_start: u32,
    pub minute_end: u32,
    pub second_start: u32,
    pub second_end: u32,
}

impl TimeRangeParams {
    /// 新規作成（バリデーション付き）
    pub fn new(
        hour_start: u32,
        hour_end: u32,
        minute_start: u32,
        minute_end: u32,
        second_start: u32,
        second_end: u32,
    ) -> Result<Self, &'static str> {
        // Hour validation
        if hour_start > 23 || hour_end > 23 {
            return Err("hour range must be within 0..=23");
        }
        if hour_start > hour_end {
            return Err("hour range start must be <= end");
        }

        // Minute validation
        if minute_start > 59 || minute_end > 59 {
            return Err("minute range must be within 0..=59");
        }
        if minute_start > minute_end {
            return Err("minute range start must be <= end");
        }

        // Second validation
        if second_start > 59 || second_end > 59 {
            return Err("second range must be within 0..=59");
        }
        if second_start > second_end {
            return Err("second range start must be <= end");
        }

        Ok(TimeRangeParams {
            hour_start,
            hour_end,
            minute_start,
            minute_end,
            second_start,
            second_end,
        })
    }

    /// 1日あたりの組み合わせ数を計算
    pub fn combos_per_day(&self) -> u32 {
        let hour_count = self.hour_end - self.hour_start + 1;
        let minute_count = self.minute_end - self.minute_start + 1;
        let second_count = self.second_end - self.second_start + 1;
        hour_count * minute_count * second_count
    }
}

/// 検索範囲パラメータ（内部型）
#[derive(Debug, Clone, Copy)]
pub struct SearchRangeParams {
    pub start_year: u32,
    pub start_month: u32,
    pub start_day: u32,
    pub start_second_offset: u32,
    pub range_seconds: u32,
}

impl SearchRangeParams {
    /// 新規作成（バリデーション付き）
    pub fn new(
        start_year: u32,
        start_month: u32,
        start_day: u32,
        start_second_offset: u32,
        range_seconds: u32,
    ) -> Result<Self, &'static str> {
        // 日付の検証
        if date_to_seconds_since_2000_local(start_year, start_month, start_day).is_none() {
            return Err("Invalid date");
        }

        if start_second_offset >= 24 * 60 * 60 {
            return Err("start_second_offset must be within a day (0-86399)");
        }

        Ok(SearchRangeParams {
            start_year,
            start_month,
            start_day,
            start_second_offset,
            range_seconds,
        })
    }

    /// 開始秒（2000年からの経過秒）を計算
    pub fn start_seconds_since_2000(&self) -> i64 {
        date_to_seconds_since_2000_local(self.start_year, self.start_month, self.start_day).unwrap()
            + self.start_second_offset as i64
    }
}

// =============================================================================
// 公開型（wasm-bindgen経由でTSに公開）
// =============================================================================

/// DS設定パラメータ（公開型）
///
/// DSハードウェア固有の設定を保持する。
/// TypeScriptから受け取り、Rust内部型に変換して使用する。
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct DSConfigJs {
    mac: [u8; 6],
    nazo: [u32; 5],
    hardware: String,
}

#[wasm_bindgen]
impl DSConfigJs {
    /// 新規作成
    #[wasm_bindgen(constructor)]
    pub fn new(mac: &[u8], nazo: &[u32], hardware: &str) -> Result<DSConfigJs, String> {
        if mac.len() != 6 {
            return Err(format!("MAC address must be 6 bytes, got {}", mac.len()));
        }
        if nazo.len() != 5 {
            return Err(format!("Nazo must be 5 elements, got {}", nazo.len()));
        }
        HardwareType::from_str(hardware).map_err(|e| e.to_string())?;

        let mut mac_arr = [0u8; 6];
        mac_arr.copy_from_slice(mac);
        let mut nazo_arr = [0u32; 5];
        nazo_arr.copy_from_slice(nazo);

        Ok(DSConfigJs {
            mac: mac_arr,
            nazo: nazo_arr,
            hardware: hardware.to_string(),
        })
    }

    #[wasm_bindgen(getter)]
    pub fn mac(&self) -> Vec<u8> {
        self.mac.to_vec()
    }

    #[wasm_bindgen(getter)]
    pub fn nazo(&self) -> Vec<u32> {
        self.nazo.to_vec()
    }

    #[wasm_bindgen(getter)]
    pub fn hardware(&self) -> String {
        self.hardware.clone()
    }
}

impl DSConfigJs {
    /// 内部型への変換
    pub fn to_ds_config(&self) -> DSConfig {
        DSConfig {
            mac: self.mac,
            nazo: self.nazo,
            hardware: HardwareType::from_str(&self.hardware).unwrap(),
        }
    }

    pub fn mac_array(&self) -> &[u8; 6] {
        &self.mac
    }

    pub fn nazo_array(&self) -> &[u32; 5] {
        &self.nazo
    }

    pub fn hardware_type(&self) -> HardwareType {
        HardwareType::from_str(&self.hardware).unwrap()
    }
}

/// セグメントパラメータ（公開型）
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub struct SegmentParamsJs {
    timer0: u32,
    vcount: u32,
    key_code: u32,
}

#[wasm_bindgen]
impl SegmentParamsJs {
    #[wasm_bindgen(constructor)]
    pub fn new(timer0: u32, vcount: u32, key_code: u32) -> SegmentParamsJs {
        SegmentParamsJs {
            timer0,
            vcount,
            key_code,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn timer0(&self) -> u32 {
        self.timer0
    }

    #[wasm_bindgen(getter)]
    pub fn vcount(&self) -> u32 {
        self.vcount
    }

    #[wasm_bindgen(getter)]
    pub fn key_code(&self) -> u32 {
        self.key_code
    }
}

impl SegmentParamsJs {
    pub fn to_segment_params(&self) -> SegmentParams {
        SegmentParams {
            timer0: self.timer0,
            vcount: self.vcount,
            key_code: self.key_code,
        }
    }
}

/// 時刻範囲パラメータ（公開型）
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub struct TimeRangeParamsJs {
    hour_start: u32,
    hour_end: u32,
    minute_start: u32,
    minute_end: u32,
    second_start: u32,
    second_end: u32,
}

#[wasm_bindgen]
impl TimeRangeParamsJs {
    #[wasm_bindgen(constructor)]
    pub fn new(
        hour_start: u32,
        hour_end: u32,
        minute_start: u32,
        minute_end: u32,
        second_start: u32,
        second_end: u32,
    ) -> Result<TimeRangeParamsJs, String> {
        TimeRangeParams::new(
            hour_start,
            hour_end,
            minute_start,
            minute_end,
            second_start,
            second_end,
        )
        .map_err(|e| e.to_string())?;

        Ok(TimeRangeParamsJs {
            hour_start,
            hour_end,
            minute_start,
            minute_end,
            second_start,
            second_end,
        })
    }

    #[wasm_bindgen(getter)]
    pub fn hour_start(&self) -> u32 {
        self.hour_start
    }

    #[wasm_bindgen(getter)]
    pub fn hour_end(&self) -> u32 {
        self.hour_end
    }

    #[wasm_bindgen(getter)]
    pub fn minute_start(&self) -> u32 {
        self.minute_start
    }

    #[wasm_bindgen(getter)]
    pub fn minute_end(&self) -> u32 {
        self.minute_end
    }

    #[wasm_bindgen(getter)]
    pub fn second_start(&self) -> u32 {
        self.second_start
    }

    #[wasm_bindgen(getter)]
    pub fn second_end(&self) -> u32 {
        self.second_end
    }
}

impl TimeRangeParamsJs {
    pub fn to_time_range_params(&self) -> TimeRangeParams {
        TimeRangeParams {
            hour_start: self.hour_start,
            hour_end: self.hour_end,
            minute_start: self.minute_start,
            minute_end: self.minute_end,
            second_start: self.second_start,
            second_end: self.second_end,
        }
    }
}

/// 検索範囲パラメータ（公開型）
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub struct SearchRangeParamsJs {
    start_year: u32,
    start_month: u32,
    start_day: u32,
    start_second_offset: u32,
    range_seconds: u32,
}

#[wasm_bindgen]
impl SearchRangeParamsJs {
    #[wasm_bindgen(constructor)]
    pub fn new(
        start_year: u32,
        start_month: u32,
        start_day: u32,
        start_second_offset: u32,
        range_seconds: u32,
    ) -> Result<SearchRangeParamsJs, String> {
        if date_to_seconds_since_2000_local(start_year, start_month, start_day).is_none() {
            return Err(format!(
                "Invalid date: {start_year}-{start_month}-{start_day}"
            ));
        }

        if start_second_offset >= 24 * 60 * 60 {
            return Err(format!("start_second_offset must be within a day, got {start_second_offset}"));
        }

        Ok(SearchRangeParamsJs {
            start_year,
            start_month,
            start_day,
            start_second_offset,
            range_seconds,
        })
    }

    #[wasm_bindgen(getter)]
    pub fn start_year(&self) -> u32 {
        self.start_year
    }

    #[wasm_bindgen(getter)]
    pub fn start_month(&self) -> u32 {
        self.start_month
    }

    #[wasm_bindgen(getter)]
    pub fn start_day(&self) -> u32 {
        self.start_day
    }

    #[wasm_bindgen(getter)]
    pub fn start_second_offset(&self) -> u32 {
        self.start_second_offset
    }

    #[wasm_bindgen(getter)]
    pub fn range_seconds(&self) -> u32 {
        self.range_seconds
    }
}

impl SearchRangeParamsJs {
    pub fn to_search_range_params(&self) -> SearchRangeParams {
        SearchRangeParams {
            start_year: self.start_year,
            start_month: self.start_month,
            start_day: self.start_day,
            start_second_offset: self.start_second_offset,
            range_seconds: self.range_seconds,
        }
    }

    pub fn start_seconds_since_2000(&self) -> i64 {
        date_to_seconds_since_2000_local(self.start_year, self.start_month, self.start_day).unwrap()
            + self.start_second_offset as i64
    }
}

// =============================================================================
// テスト
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_type_from_str() {
        assert_eq!(HardwareType::from_str("DS").unwrap(), HardwareType::DS);
        assert_eq!(
            HardwareType::from_str("DS_LITE").unwrap(),
            HardwareType::DSLite
        );
        assert_eq!(
            HardwareType::from_str("3DS").unwrap(),
            HardwareType::ThreeDS
        );
        assert!(HardwareType::from_str("Invalid").is_err());
    }

    #[test]
    fn test_hardware_type_frame() {
        assert_eq!(HardwareType::DS.frame(), 8);
        assert_eq!(HardwareType::DSLite.frame(), 6);
        assert_eq!(HardwareType::ThreeDS.frame(), 9);
    }

    #[test]
    fn test_hardware_type_as_str() {
        assert_eq!(HardwareType::DS.as_str(), "DS");
        assert_eq!(HardwareType::DSLite.as_str(), "DS_LITE");
        assert_eq!(HardwareType::ThreeDS.as_str(), "3DS");
    }

    #[test]
    fn test_ds_config() {
        let mac = [0x00, 0x09, 0xBF, 0xAA, 0xBB, 0xCC];
        let nazo = [0x02215F10, 0x02215F30, 0x02215F20, 0x02761008, 0x00000000];
        let config = DSConfig::new(mac, nazo, HardwareType::DS);

        assert_eq!(config.mac, mac);
        assert_eq!(config.nazo, nazo);
        assert_eq!(config.hardware, HardwareType::DS);
        assert_eq!(config.frame(), 8);
    }

    #[test]
    fn test_segment_params() {
        let params = SegmentParams::new(0x1000, 0x60, 0x2FFF);
        assert_eq!(params.timer0, 0x1000);
        assert_eq!(params.vcount, 0x60);
        assert_eq!(params.key_code, 0x2FFF);
    }

    #[test]
    fn test_time_range_params() {
        let config = TimeRangeParams::new(10, 12, 0, 59, 0, 59).unwrap();
        assert_eq!(config.combos_per_day(), 3 * 60 * 60);
    }

    #[test]
    fn test_time_range_params_validation() {
        assert!(TimeRangeParams::new(24, 25, 0, 59, 0, 59).is_err());
        assert!(TimeRangeParams::new(12, 10, 0, 59, 0, 59).is_err());
    }

    #[test]
    fn test_search_range_params() {
        let params = SearchRangeParams::new(2024, 1, 15, 86400).unwrap();
        assert_eq!(params.start_year, 2024);
        assert_eq!(params.start_month, 1);
        assert_eq!(params.start_day, 15);
        assert_eq!(params.range_seconds, 86400);
        assert!(params.start_seconds_since_2000() > 0);
    }

    #[test]
    fn test_search_range_params_invalid() {
        assert!(SearchRangeParams::new(2024, 13, 1, 86400).is_err());
        assert!(SearchRangeParams::new(2024, 2, 30, 86400).is_err());
    }

    #[test]
    fn test_ds_config_js_valid() {
        let mac = [0x00, 0x09, 0xBF, 0xAA, 0xBB, 0xCC];
        let nazo = [0x02215F10, 0x02215F30, 0x02215F20, 0x02761008, 0x00000000];
        let config = DSConfigJs::new(&mac, &nazo, "DS").unwrap();

        assert_eq!(config.mac_array(), &mac);
        assert_eq!(config.nazo_array(), &nazo);
        assert_eq!(config.hardware_type(), HardwareType::DS);
    }

    #[test]
    fn test_ds_config_js_invalid_mac() {
        let mac = [0x00, 0x09, 0xBF, 0xAA];
        let nazo = [0x02215F10, 0x02215F30, 0x02215F20, 0x02761008, 0x00000000];
        assert!(DSConfigJs::new(&mac, &nazo, "DS").is_err());
    }

    #[test]
    fn test_ds_config_js_invalid_nazo() {
        let mac = [0x00, 0x09, 0xBF, 0xAA, 0xBB, 0xCC];
        let nazo = [0x02215F10, 0x02215F30];
        assert!(DSConfigJs::new(&mac, &nazo, "DS").is_err());
    }

    #[test]
    fn test_ds_config_js_invalid_hardware() {
        let mac = [0x00, 0x09, 0xBF, 0xAA, 0xBB, 0xCC];
        let nazo = [0x02215F10, 0x02215F30, 0x02215F20, 0x02761008, 0x00000000];
        assert!(DSConfigJs::new(&mac, &nazo, "Invalid").is_err());
    }

    #[test]
    fn test_segment_params_js() {
        let params = SegmentParamsJs::new(0x1000, 0x60, 0x2FFF);
        assert_eq!(params.timer0(), 0x1000);
        assert_eq!(params.vcount(), 0x60);
        assert_eq!(params.key_code(), 0x2FFF);
    }

    #[test]
    fn test_time_range_params_js_valid() {
        let params = TimeRangeParamsJs::new(10, 12, 0, 59, 0, 59).unwrap();
        assert_eq!(params.hour_start(), 10);
        assert_eq!(params.hour_end(), 12);

        let config = params.to_time_range_params();
        assert_eq!(config.hour_start, 10);
        assert_eq!(config.hour_end, 12);
    }

    #[test]
    fn test_time_range_params_js_invalid() {
        assert!(TimeRangeParamsJs::new(24, 25, 0, 59, 0, 59).is_err());
        assert!(TimeRangeParamsJs::new(12, 10, 0, 59, 0, 59).is_err());
    }

    #[test]
    fn test_search_range_params_js_valid() {
        let params = SearchRangeParamsJs::new(2024, 1, 15, 3600, 86400).unwrap();
        assert_eq!(params.start_year(), 2024);
        assert_eq!(params.start_month(), 1);
        assert_eq!(params.start_day(), 15);
        assert_eq!(params.start_second_offset(), 3600);
        assert_eq!(params.range_seconds(), 86400);
        assert!(params.start_seconds_since_2000() > 0);
    }

    #[test]
    fn test_search_range_params_js_invalid_date() {
        assert!(SearchRangeParamsJs::new(2024, 13, 1, 0, 86400).is_err());
        assert!(SearchRangeParamsJs::new(2024, 2, 30, 0, 86400).is_err());
    }
}

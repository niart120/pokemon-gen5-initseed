//! 検索処理共通モジュール
//!
//! 起動時間検索で共通利用される定数・型・ユーティリティ関数を提供する。
//!
//! ## モジュール構成
//!
//! ### 内部型（Rust内部のみ）
//! - `HardwareType`: ハードウェア種別enum
//! - `DSConfig`: DS設定パラメータ
//! - `SegmentParams`: セグメントパラメータ
//! - `TimeRangeParams`: 時刻範囲パラメータ
//! - `SearchRangeParams`: 検索範囲パラメータ
//! - `BaseMessageBuilder`: SHA-1メッセージ構築
//! - `DateTimeCodeCalculator`: 日時コード計算
//! - `HashValues`: SHA-1ハッシュ値
//!
//! ### 公開型（wasm-bindgen経由でTSに公開）
//! - `DSConfigJs`: DS設定パラメータ（→ DSConfig に変換）
//! - `SegmentParamsJs`: セグメントパラメータ（→ SegmentParams に変換）
//! - `TimeRangeParamsJs`: 時刻範囲パラメータ（→ TimeRangeParams に変換）
//! - `SearchRangeParamsJs`: 検索範囲パラメータ（→ SearchRangeParams に変換）

use crate::datetime_codes::{DateCodeGenerator, TimeCodeGenerator};
use crate::sha1::swap_bytes_32;
use chrono::{Datelike, NaiveDate, Timelike};
use wasm_bindgen::prelude::*;

// =============================================================================
// 定数
// =============================================================================

/// 2000年1月1日 00:00:00 UTCのUnix時間
pub const EPOCH_2000_UNIX: i64 = 946684800;

/// 1日の秒数
pub const SECONDS_PER_DAY: i64 = 86_400;

/// Hardware別のframe値
pub const HARDWARE_FRAME_DS: u32 = 8;
pub const HARDWARE_FRAME_DS_LITE: u32 = 6;
pub const HARDWARE_FRAME_3DS: u32 = 9;

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
    pub fn new(
        mac: [u8; 6],
        nazo: [u32; 5],
        hardware: HardwareType,
    ) -> Self {
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
        Self { timer0, vcount, key_code }
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
    pub range_seconds: u32,
}

impl SearchRangeParams {
    /// 新規作成（バリデーション付き）
    pub fn new(
        start_year: u32,
        start_month: u32,
        start_day: u32,
        range_seconds: u32,
    ) -> Result<Self, &'static str> {
        // 日付の検証
        if date_to_seconds_since_2000(start_year, start_month, start_day).is_none() {
            return Err("Invalid date");
        }

        Ok(SearchRangeParams {
            start_year,
            start_month,
            start_day,
            range_seconds,
        })
    }

    /// 開始秒（2000年からの経過秒）を計算
    pub fn start_seconds_since_2000(&self) -> i64 {
        date_to_seconds_since_2000(self.start_year, self.start_month, self.start_day).unwrap()
    }
}

/// 許可秒マスクを構築
///
/// 86,400要素の配列で、各インデックスが1日の秒数（0-86399）に対応し、
/// 検索対象とする秒かどうかを示す。
pub fn build_allowed_second_mask(range: &TimeRangeParams) -> Box<[bool; 86400]> {
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

// =============================================================================
// 基本メッセージ構築
// =============================================================================

/// SHA-1計算用の基本メッセージビルダー
///
/// MAC/Nazo/Frame/Timer0/VCount/KeyCode など固定パラメータから base_message を構築する。
/// セグメント（timer0, vcount, key_code）は構築時に固定される。
#[derive(Debug, Clone)]
pub struct BaseMessageBuilder {
    base_message: [u32; 16],
}

impl BaseMessageBuilder {
    /// 新規作成（内部型から構築）
    ///
    /// # Arguments
    /// - `ds_config`: DS設定パラメータ
    /// - `segment`: セグメントパラメータ
    pub fn from_params(ds_config: &DSConfig, segment: &SegmentParams) -> Self {
        Self::new_internal(
            &ds_config.mac,
            &ds_config.nazo,
            ds_config.frame(),
            segment.timer0,
            segment.vcount,
            segment.key_code,
        )
    }

    /// 新規作成（プリミティブ引数版、バリデーション付き）
    ///
    /// # Arguments
    /// - `mac`: MACアドレス（6バイト）
    /// - `nazo`: Nazo値（5ワード）
    /// - `frame`: Frame値（Hardware依存）
    /// - `timer0`: Timer0値
    /// - `vcount`: VCount値
    /// - `key_code`: キーコード
    pub fn new(
        mac: &[u8],
        nazo: &[u32],
        frame: u32,
        timer0: u32,
        vcount: u32,
        key_code: u32,
    ) -> Result<Self, &'static str> {
        if mac.len() != 6 {
            return Err("MAC address must be 6 bytes");
        }
        if nazo.len() != 5 {
            return Err("nazo must be 5 32-bit words");
        }

        let mut mac_arr = [0u8; 6];
        mac_arr.copy_from_slice(mac);
        let mut nazo_arr = [0u32; 5];
        nazo_arr.copy_from_slice(nazo);

        Ok(Self::new_internal(&mac_arr, &nazo_arr, frame, timer0, vcount, key_code))
    }

    fn new_internal(
        mac: &[u8; 6],
        nazo: &[u32; 5],
        frame: u32,
        timer0: u32,
        vcount: u32,
        key_code: u32,
    ) -> Self {

        let mut base_message = [0u32; 16];

        // data[0-4]: Nazo values (little-endian conversion)
        for i in 0..5 {
            base_message[i] = swap_bytes_32(nazo[i]);
        }

        // data[5]: (VCount << 16) | Timer0
        base_message[5] = swap_bytes_32((vcount << 16) | timer0);

        // data[6]: MAC address lower 16 bits (no endian conversion)
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

        // data[8]: Date - 動的に設定
        // data[9]: Time - 動的に設定
        // data[10-11]: Fixed values
        base_message[10] = 0x00000000;
        base_message[11] = 0x00000000;

        // data[12]: Key input
        base_message[12] = swap_bytes_32(key_code);

        // data[13-15]: SHA-1 padding
        base_message[13] = 0x80000000;
        base_message[14] = 0x00000000;
        base_message[15] = 0x000001A0;

        Self { base_message }
    }

    /// 基本メッセージを取得
    pub fn base_message(&self) -> &[u32; 16] {
        &self.base_message
    }

    /// 日時コードを適用したメッセージを構築
    #[inline(always)]
    pub fn build_message(&self, date_code: u32, time_code: u32) -> [u32; 16] {
        let mut message = self.base_message;
        message[8] = date_code;
        message[9] = time_code;
        message
    }
}

// =============================================================================
// 日時コード計算
// =============================================================================

/// 日時コード計算器
///
/// seconds_since_2000 から日時コード (time_code, date_code) を計算する。
pub struct DateTimeCodeCalculator<'a> {
    allowed_second_mask: &'a [bool; 86400],
    hardware: HardwareType,
}

impl<'a> DateTimeCodeCalculator<'a> {
    pub fn new(allowed_second_mask: &'a [bool; 86400], hardware: HardwareType) -> Self {
        Self {
            allowed_second_mask,
            hardware,
        }
    }

    /// 日時コード生成
    ///
    /// 許可範囲外の秒の場合は None を返す。
    #[inline(always)]
    pub fn calculate(&self, seconds_since_2000: i64) -> Option<(u32, u32)> {
        if seconds_since_2000 < 0 {
            return None;
        }

        let seconds_of_day = (seconds_since_2000 % SECONDS_PER_DAY) as u32;
        if !self.is_second_allowed(seconds_of_day) {
            return None;
        }

        let date_index = (seconds_since_2000 / SECONDS_PER_DAY) as u32;

        let time_code =
            TimeCodeGenerator::get_time_code_for_hardware(seconds_of_day, self.hardware.as_str());
        let date_code = DateCodeGenerator::get_date_code(date_index);

        Some((time_code, date_code))
    }

    #[inline(always)]
    fn is_second_allowed(&self, second_of_day: u32) -> bool {
        self.allowed_second_mask[second_of_day as usize]
    }
}

// =============================================================================
// 日時表示生成
// =============================================================================

/// 結果表示用の日時を生成
///
/// seconds_since_2000 から (year, month, day, hour, minute, second) を生成する。
pub fn generate_display_datetime(
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

/// 開始日時をseconds_since_2000に変換
pub fn datetime_to_seconds_since_2000(
    year: u32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
) -> Option<i64> {
    let datetime =
        NaiveDate::from_ymd_opt(year as i32, month, day)?.and_hms_opt(hour, minute, second)?;
    let unix = datetime.and_utc().timestamp();
    Some(unix - EPOCH_2000_UNIX)
}

/// 開始日時（時刻0:0:0）をseconds_since_2000に変換
pub fn date_to_seconds_since_2000(year: u32, month: u32, day: u32) -> Option<i64> {
    datetime_to_seconds_since_2000(year, month, day, 0, 0, 0)
}

// =============================================================================
// SHA-1 ハッシュ値構造体
// =============================================================================

/// SHA-1ハッシュ値（5ワード）
#[derive(Debug, Clone, Copy)]
pub struct HashValues {
    pub h0: u32,
    pub h1: u32,
    pub h2: u32,
    pub h3: u32,
    pub h4: u32,
}

impl HashValues {
    pub fn new(h0: u32, h1: u32, h2: u32, h3: u32, h4: u32) -> Self {
        Self { h0, h1, h2, h3, h4 }
    }

    /// 16進数文字列に変換
    pub fn to_hex_string(&self) -> String {
        format!(
            "{:08x}{:08x}{:08x}{:08x}{:08x}",
            self.h0, self.h1, self.h2, self.h3, self.h4
        )
    }

    /// 64bit LCG Seedを計算
    #[inline]
    pub fn to_lcg_seed(&self) -> u64 {
        let h0_le = swap_bytes_32(self.h0) as u64;
        let h1_le = swap_bytes_32(self.h1) as u64;
        (h1_le << 32) | h0_le
    }

    /// 32bit MT Seedを計算（ポケモンBW/BW2用）
    #[inline]
    pub fn to_mt_seed(&self) -> u32 {
        crate::sha1::calculate_pokemon_seed_from_hash(self.h0, self.h1)
    }
}

// =============================================================================
// 公開Value Object（wasm-bindgen経由でTSに公開）
// =============================================================================

/// DS設定パラメータ
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
    ///
    /// # Arguments
    /// * `mac` - MACアドレス（6バイト）
    /// * `nazo` - Nazo値（5要素）
    /// * `hardware` - ハードウェア種別 ("DS", "DS_LITE", "3DS")
    #[wasm_bindgen(constructor)]
    pub fn new(mac: &[u8], nazo: &[u32], hardware: &str) -> Result<DSConfigJs, String> {
        if mac.len() != 6 {
            return Err(format!("MAC address must be 6 bytes, got {}", mac.len()));
        }
        if nazo.len() != 5 {
            return Err(format!("Nazo must be 5 elements, got {}", nazo.len()));
        }
        // ハードウェア種別の検証
        HardwareType::from_str(hardware)
            .map_err(|e| e.to_string())?;

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

    /// MACアドレス取得
    #[wasm_bindgen(getter)]
    pub fn mac(&self) -> Vec<u8> {
        self.mac.to_vec()
    }

    /// Nazo値取得
    #[wasm_bindgen(getter)]
    pub fn nazo(&self) -> Vec<u32> {
        self.nazo.to_vec()
    }

    /// ハードウェア種別取得
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

    /// MACアドレス配列への参照
    pub fn mac_array(&self) -> &[u8; 6] {
        &self.mac
    }

    /// Nazo配列への参照
    pub fn nazo_array(&self) -> &[u32; 5] {
        &self.nazo
    }

    /// ハードウェア種別を取得
    pub fn hardware_type(&self) -> HardwareType {
        HardwareType::from_str(&self.hardware).unwrap()
    }
}

/// セグメントパラメータ
///
/// Timer0/VCount/KeyCodeの組み合わせを保持する。
/// TypeScriptで生成し、検索イテレータに渡す。
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub struct SegmentParamsJs {
    timer0: u32,
    vcount: u32,
    key_code: u32,
}

#[wasm_bindgen]
impl SegmentParamsJs {
    /// 新規作成
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
    /// 内部型への変換
    pub fn to_segment_params(&self) -> SegmentParams {
        SegmentParams {
            timer0: self.timer0,
            vcount: self.vcount,
            key_code: self.key_code,
        }
    }
}

/// 時刻範囲パラメータ
///
/// 1日内の許可時刻範囲を指定する。
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
    /// 新規作成
    ///
    /// # Arguments
    /// * `hour_start` - 開始時 (0-23)
    /// * `hour_end` - 終了時 (0-23)
    /// * `minute_start` - 開始分 (0-59)
    /// * `minute_end` - 終了分 (0-59)
    /// * `second_start` - 開始秒 (0-59)
    /// * `second_end` - 終了秒 (0-59)
    #[wasm_bindgen(constructor)]
    pub fn new(
        hour_start: u32,
        hour_end: u32,
        minute_start: u32,
        minute_end: u32,
        second_start: u32,
        second_end: u32,
    ) -> Result<TimeRangeParamsJs, String> {
        // 内部型で検証
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
    /// 内部型への変換
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

/// 検索範囲パラメータ
///
/// 検索開始日と範囲（秒数）を指定する。
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub struct SearchRangeParamsJs {
    start_year: u32,
    start_month: u32,
    start_day: u32,
    range_seconds: u32,
}

#[wasm_bindgen]
impl SearchRangeParamsJs {
    /// 新規作成
    ///
    /// # Arguments
    /// * `start_year` - 開始年 (2000-2099)
    /// * `start_month` - 開始月 (1-12)
    /// * `start_day` - 開始日 (1-31)
    /// * `range_seconds` - 検索範囲（秒数）
    #[wasm_bindgen(constructor)]
    pub fn new(
        start_year: u32,
        start_month: u32,
        start_day: u32,
        range_seconds: u32,
    ) -> Result<SearchRangeParamsJs, String> {
        // 日付の検証
        if date_to_seconds_since_2000(start_year, start_month, start_day).is_none() {
            return Err(format!(
                "Invalid date: {}-{}-{}",
                start_year, start_month, start_day
            ));
        }

        Ok(SearchRangeParamsJs {
            start_year,
            start_month,
            start_day,
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
    pub fn range_seconds(&self) -> u32 {
        self.range_seconds
    }
}

impl SearchRangeParamsJs {
    /// 内部型への変換
    pub fn to_search_range_params(&self) -> SearchRangeParams {
        SearchRangeParams {
            start_year: self.start_year,
            start_month: self.start_month,
            start_day: self.start_day,
            range_seconds: self.range_seconds,
        }
    }

    /// 開始秒（2000年からの経過秒）を計算
    pub fn start_seconds_since_2000(&self) -> i64 {
        date_to_seconds_since_2000(self.start_year, self.start_month, self.start_day).unwrap()
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

    // =========================================================================
    // 内部型テスト
    // =========================================================================

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
    fn test_base_message_builder_from_params() {
        let mac = [0x00, 0x09, 0xBF, 0xAA, 0xBB, 0xCC];
        let nazo = [0x02215F10, 0x02215F30, 0x02215F20, 0x02761008, 0x00000000];
        let ds_config = DSConfig::new(mac, nazo, HardwareType::DS);
        let segment = SegmentParams::new(0x1000, 0x60, 0x2FFF);

        let builder = BaseMessageBuilder::from_params(&ds_config, &segment);
        let base = builder.base_message();

        // Check padding values
        assert_eq!(base[13], 0x80000000);
        assert_eq!(base[14], 0x00000000);
        assert_eq!(base[15], 0x000001A0);
    }

    #[test]
    fn test_time_range_params() {
        let config = TimeRangeParams::new(10, 12, 0, 59, 0, 59).unwrap();
        assert_eq!(config.combos_per_day(), 3 * 60 * 60);
    }

    #[test]
    fn test_time_range_params_validation() {
        // Invalid hour range
        assert!(TimeRangeParams::new(24, 25, 0, 59, 0, 59).is_err());
        // Start > End
        assert!(TimeRangeParams::new(12, 10, 0, 59, 0, 59).is_err());
    }

    #[test]
    fn test_build_allowed_second_mask() {
        let config = TimeRangeParams {
            hour_start: 10,
            hour_end: 10,
            minute_start: 30,
            minute_end: 30,
            second_start: 0,
            second_end: 59,
        };
        let mask = build_allowed_second_mask(&config);

        // 10:30:00 should be allowed
        let idx_10_30_00 = 10 * 3600 + 30 * 60 + 0;
        assert!(mask[idx_10_30_00]);

        // 10:31:00 should not be allowed
        let idx_10_31_00 = 10 * 3600 + 31 * 60 + 0;
        assert!(!mask[idx_10_31_00]);
    }

    #[test]
    fn test_combos_per_day() {
        let config = TimeRangeParams {
            hour_start: 10,
            hour_end: 12,
            minute_start: 0,
            minute_end: 59,
            second_start: 0,
            second_end: 59,
        };
        assert_eq!(config.combos_per_day(), 3 * 60 * 60);
    }

    #[test]
    fn test_base_message_builder() {
        let mac = [0x00, 0x09, 0xBF, 0xAA, 0xBB, 0xCC];
        let nazo = [0x02215F10, 0x02215F30, 0x02215F20, 0x02761008, 0x00000000];
        let frame = 8;
        let timer0 = 0x1000;
        let vcount = 0x60;
        let key_code = 0x2FFF;

        let builder =
            BaseMessageBuilder::new(&mac, &nazo, frame, timer0, vcount, key_code).unwrap();
        let base = builder.base_message();

        // Check padding values
        assert_eq!(base[13], 0x80000000);
        assert_eq!(base[14], 0x00000000);
        assert_eq!(base[15], 0x000001A0);

        // Check that timer0/vcount are set
        assert_ne!(base[5], 0);

        // Check that key_code is set
        assert_ne!(base[12], 0);
    }

    #[test]
    fn test_base_message_builder_build_message() {
        let mac = [0x00, 0x09, 0xBF, 0xAA, 0xBB, 0xCC];
        let nazo = [0x02215F10, 0x02215F30, 0x02215F20, 0x02761008, 0x00000000];
        let frame = 8;

        let builder = BaseMessageBuilder::new(&mac, &nazo, frame, 0x1000, 0x60, 0x2FFF).unwrap();
        let message = builder.build_message(0x12345678, 0xABCDEF00);

        assert_eq!(message[8], 0x12345678);
        assert_eq!(message[9], 0xABCDEF00);
    }

    #[test]
    fn test_hash_values() {
        let hash = HashValues::new(0x12345678, 0xABCDEF01, 0x11111111, 0x22222222, 0x33333333);
        assert_eq!(
            hash.to_hex_string(),
            "12345678abcdef01111111112222222233333333"
        );
    }

    #[test]
    fn test_hash_values_to_lcg_seed() {
        let h0: u32 = 0x12345678;
        let h1: u32 = 0xABCDEF01;
        let hash = HashValues::new(h0, h1, 0, 0, 0);
        let seed = hash.to_lcg_seed();

        let h0_le = swap_bytes_32(h0) as u64;
        let h1_le = swap_bytes_32(h1) as u64;
        let expected = (h1_le << 32) | h0_le;
        assert_eq!(seed, expected);
    }

    #[test]
    fn test_datetime_to_seconds_since_2000() {
        // 2000-01-01 00:00:00 should be 0
        let seconds = datetime_to_seconds_since_2000(2000, 1, 1, 0, 0, 0).unwrap();
        assert_eq!(seconds, 0);

        // 2000-01-01 00:00:01 should be 1
        let seconds = datetime_to_seconds_since_2000(2000, 1, 1, 0, 0, 1).unwrap();
        assert_eq!(seconds, 1);

        // 2000-01-02 00:00:00 should be 86400
        let seconds = datetime_to_seconds_since_2000(2000, 1, 2, 0, 0, 0).unwrap();
        assert_eq!(seconds, 86400);
    }

    #[test]
    fn test_generate_display_datetime() {
        // 0 seconds since 2000 -> 2000-01-01 00:00:00
        let (year, month, day, hour, minute, second) = generate_display_datetime(0).unwrap();
        assert_eq!(
            (year, month, day, hour, minute, second),
            (2000, 1, 1, 0, 0, 0)
        );

        // 86400 seconds since 2000 -> 2000-01-02 00:00:00
        let (year, month, day, hour, minute, second) = generate_display_datetime(86400).unwrap();
        assert_eq!(
            (year, month, day, hour, minute, second),
            (2000, 1, 2, 0, 0, 0)
        );
    }

    // =========================================================================
    // 公開Value Objectテスト
    // =========================================================================

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
        let mac = [0x00, 0x09, 0xBF, 0xAA]; // 4 bytes instead of 6
        let nazo = [0x02215F10, 0x02215F30, 0x02215F20, 0x02761008, 0x00000000];
        assert!(DSConfigJs::new(&mac, &nazo, "DS").is_err());
    }

    #[test]
    fn test_ds_config_js_invalid_nazo() {
        let mac = [0x00, 0x09, 0xBF, 0xAA, 0xBB, 0xCC];
        let nazo = [0x02215F10, 0x02215F30]; // 2 elements instead of 5
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
        // Invalid hour
        assert!(TimeRangeParamsJs::new(24, 25, 0, 59, 0, 59).is_err());
        // Start > End
        assert!(TimeRangeParamsJs::new(12, 10, 0, 59, 0, 59).is_err());
    }

    #[test]
    fn test_search_range_params_js_valid() {
        let params = SearchRangeParamsJs::new(2024, 1, 15, 86400).unwrap();
        assert_eq!(params.start_year(), 2024);
        assert_eq!(params.start_month(), 1);
        assert_eq!(params.start_day(), 15);
        assert_eq!(params.range_seconds(), 86400);

        // Verify start_seconds_since_2000
        let seconds = params.start_seconds_since_2000();
        assert!(seconds > 0);
    }

    #[test]
    fn test_search_range_params_js_invalid_date() {
        // Invalid month
        assert!(SearchRangeParamsJs::new(2024, 13, 1, 86400).is_err());
        // Invalid day
        assert!(SearchRangeParamsJs::new(2024, 2, 30, 86400).is_err());
    }
}

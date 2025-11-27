//! 検索処理共通モジュール
//!
//! `integrated_search` と `egg_boot_timing_search` で共通利用される
//! 定数・型・ユーティリティ関数を提供する。

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
    pub fn from_str(hardware: &str) -> Result<Self, JsValue> {
        match hardware {
            "DS" => Ok(HardwareType::DS),
            "DS_LITE" => Ok(HardwareType::DSLite),
            "3DS" => Ok(HardwareType::ThreeDS),
            _ => Err(JsValue::from_str("Hardware must be DS, DS_LITE, or 3DS")),
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
// 日時範囲設定
// =============================================================================

/// 日次時刻範囲設定
///
/// 1日の中で検索対象とする時刻範囲を指定する。
#[derive(Debug, Clone, Copy)]
pub struct DailyTimeRangeConfig {
    pub hour_start: u32,
    pub hour_end: u32,
    pub minute_start: u32,
    pub minute_end: u32,
    pub second_start: u32,
    pub second_end: u32,
}

impl DailyTimeRangeConfig {
    /// 新規作成（バリデーション付き）
    pub fn new(
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

    /// 1日あたりの組み合わせ数を計算
    pub fn combos_per_day(&self) -> u32 {
        let hour_count = self.hour_end - self.hour_start + 1;
        let minute_count = self.minute_end - self.minute_start + 1;
        let second_count = self.second_end - self.second_start + 1;
        hour_count * minute_count * second_count
    }
}

/// 許可秒マスクを構築
///
/// 86,400要素の配列で、各インデックスが1日の秒数（0-86399）に対応し、
/// 検索対象とする秒かどうかを示す。
pub fn build_allowed_second_mask(range: &DailyTimeRangeConfig) -> Box<[bool; 86400]> {
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
// キーコード生成
// =============================================================================

/// 実機上で不可能なキー入力の組み合わせをチェックする
///
/// XOR 0x2FFFする前の生のキーコード（押されているキーが1のビット）について判定
///
/// # 不可能な組み合わせ
/// - 上下同時押し (UP: bit 6, DOWN: bit 7)
/// - 左右同時押し (LEFT: bit 5, RIGHT: bit 4)
/// - Start, Select, L, R の4つ同時押し (START: bit 3, SELECT: bit 2, L: bit 9, R: bit 8)
fn is_invalid_key_combination(raw_key_input: u32) -> bool {
    const RIGHT: u32 = 1 << 4;
    const LEFT: u32 = 1 << 5;
    const UP: u32 = 1 << 6;
    const DOWN: u32 = 1 << 7;
    const R: u32 = 1 << 8;
    const L: u32 = 1 << 9;
    const SELECT: u32 = 1 << 2;
    const START: u32 = 1 << 3;

    // 上下同時押しチェック
    if (raw_key_input & UP) != 0 && (raw_key_input & DOWN) != 0 {
        return true;
    }

    // 左右同時押しチェック
    if (raw_key_input & LEFT) != 0 && (raw_key_input & RIGHT) != 0 {
        return true;
    }

    // Start, Select, L, R の4つ同時押しチェック
    if (raw_key_input & START) != 0
        && (raw_key_input & SELECT) != 0
        && (raw_key_input & L) != 0
        && (raw_key_input & R) != 0
    {
        return true;
    }

    false
}

/// キーマスクからべき集合を生成し、各要素を 0x2FFF と XOR したキーコードを返す
///
/// 実機上で不可能なキー入力の組み合わせは除外される。
pub fn generate_key_codes(key_input_mask: u32) -> Vec<u32> {
    let mut enabled_bits = Vec::new();

    // 有効なビット位置を収集
    for bit in 0..12 {
        if (key_input_mask & (1 << bit)) != 0 {
            enabled_bits.push(bit);
        }
    }

    // べき集合を生成（2^n 通りの組み合わせ）
    let n = enabled_bits.len();
    let total_combinations = 1 << n; // 2^n
    let mut key_codes = Vec::with_capacity(total_combinations);

    for i in 0..total_combinations {
        let mut combination = 0u32;
        for (bit_index, &bit_pos) in enabled_bits.iter().enumerate() {
            if (i & (1 << bit_index)) != 0 {
                combination |= 1 << bit_pos;
            }
        }

        // 不可能なキー入力の組み合わせをチェック
        if is_invalid_key_combination(combination) {
            continue;
        }

        // 押されているビットが1、押されていないビットが0の状態を0x2FFFとXOR
        let key_code = combination ^ 0x2FFF;
        key_codes.push(key_code);
    }

    key_codes
}

// =============================================================================
// セグメントパラメータ
// =============================================================================

/// 固定セグメントパラメータ
///
/// TypeScript側のセグメントループから渡される固定値。
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

// =============================================================================
// 基本メッセージ構築
// =============================================================================

/// SHA-1計算用の基本メッセージビルダー
///
/// MAC/Nazo/Frame など固定パラメータから base_message を構築する。
#[derive(Debug, Clone)]
pub struct BaseMessageBuilder {
    base_message: [u32; 16],
}

impl BaseMessageBuilder {
    /// 新規作成
    ///
    /// # Arguments
    /// - `mac`: MACアドレス（6バイト）
    /// - `nazo`: Nazo値（5ワード）
    /// - `frame`: Frame値（Hardware依存）
    pub fn new(mac: &[u8], nazo: &[u32], frame: u32) -> Result<Self, JsValue> {
        if mac.len() != 6 {
            return Err(JsValue::from_str("MAC address must be 6 bytes"));
        }
        if nazo.len() != 5 {
            return Err(JsValue::from_str("nazo must be 5 32-bit words"));
        }

        let mut base_message = [0u32; 16];

        // data[0-4]: Nazo values (little-endian conversion)
        for i in 0..5 {
            base_message[i] = swap_bytes_32(nazo[i]);
        }

        // data[5]: (VCount << 16) | Timer0 - 動的に設定（ここでは0）
        base_message[5] = 0;

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

        // data[12]: Key input - 動的に設定
        base_message[12] = 0;

        // data[13-15]: SHA-1 padding
        base_message[13] = 0x80000000;
        base_message[14] = 0x00000000;
        base_message[15] = 0x000001A0;

        Ok(Self { base_message })
    }

    /// 基本メッセージを取得
    pub fn base_message(&self) -> &[u32; 16] {
        &self.base_message
    }

    /// セグメントパラメータと日時コードを適用したメッセージを構築
    #[inline(always)]
    pub fn build_message(
        &self,
        segment: &SegmentParams,
        date_code: u32,
        time_code: u32,
    ) -> [u32; 16] {
        let mut message = self.base_message;
        message[5] = swap_bytes_32((segment.vcount << 16) | segment.timer0);
        message[8] = date_code;
        message[9] = time_code;
        message[12] = swap_bytes_32(segment.key_code);
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
    let datetime = NaiveDate::from_ymd_opt(year as i32, month, day)?
        .and_hms_opt(hour, minute, second)?;
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
}

// =============================================================================
// LCG Seed 計算
// =============================================================================

/// SHA-1ハッシュ値から64bit LCG Seedを計算
#[inline]
pub fn calculate_lcg_seed_from_hash(h0: u32, h1: u32) -> u64 {
    let h0_le = swap_bytes_32(h0) as u64;
    let h1_le = swap_bytes_32(h1) as u64;
    (h1_le << 32) | h0_le
}

// =============================================================================
// テスト
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Tests involving JsValue (like HardwareType::from_str, DailyTimeRangeConfig::new)
    // are skipped in native test environment as JsValue operations panic outside WASM.
    // These are tested via wasm-bindgen-test in browser environment.

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
    fn test_build_allowed_second_mask_direct() {
        // Manually create config to avoid JsValue
        let config = DailyTimeRangeConfig {
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
        let config = DailyTimeRangeConfig {
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
    fn test_generate_key_codes() {
        // No keys enabled
        let codes = generate_key_codes(0);
        assert_eq!(codes.len(), 1);
        assert_eq!(codes[0], 0x2FFF);

        // One key enabled (bit 0)
        let codes = generate_key_codes(0b1);
        assert_eq!(codes.len(), 2);
    }

    #[test]
    fn test_generate_key_codes_invalid_combinations() {
        // UP + DOWN enabled (should exclude simultaneous press)
        let codes = generate_key_codes(0b11000000); // bit 6 and 7
        // Should have: none pressed, UP only, DOWN only (not both)
        assert_eq!(codes.len(), 3);
    }

    #[test]
    fn test_segment_params() {
        let segment = SegmentParams::new(0x1000, 0x60, 0x2FFF);
        assert_eq!(segment.timer0, 0x1000);
        assert_eq!(segment.vcount, 0x60);
        assert_eq!(segment.key_code, 0x2FFF);
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
    fn test_calculate_lcg_seed_from_hash() {
        let h0: u32 = 0x12345678;
        let h1: u32 = 0xABCDEF01;
        let seed = calculate_lcg_seed_from_hash(h0, h1);

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
}

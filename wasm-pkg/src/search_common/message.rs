//! メッセージ構築モジュール
//!
//! SHA-1計算用のメッセージ構築を担当する。

use super::params::{DSConfig, SegmentParams};
use crate::utils::EndianUtils;

/// ローカルヘルパー関数（EndianUtils経由）
#[inline(always)]
fn swap_bytes_32(value: u32) -> u32 {
    EndianUtils::swap_bytes_32(value)
}

// =============================================================================
// 基本メッセージビルダー
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

        Ok(Self::new_internal(
            &mac_arr, &nazo_arr, frame, timer0, vcount, key_code,
        ))
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

    /// 4組分のメッセージバッファをbase_messageで初期化
    ///
    /// SIMD処理前に1回だけ呼び出し、その後は`write_datetime_only`で
    /// date/timeのみを書き換えることで、不要なコピーを削減する。
    #[inline(always)]
    pub fn init_message_buffer(&self, buffer: &mut [u32; 64]) {
        buffer[0..16].copy_from_slice(&self.base_message);
        buffer[16..32].copy_from_slice(&self.base_message);
        buffer[32..48].copy_from_slice(&self.base_message);
        buffer[48..64].copy_from_slice(&self.base_message);
    }

    /// 日時コードを適用したメッセージを構築
    #[inline(always)]
    pub fn build_message(&self, date_code: u32, time_code: u32) -> [u32; 16] {
        let mut message = self.base_message;
        message[8] = date_code;
        message[9] = time_code;
        message
    }

    /// バッファの指定位置にdate/timeのみを書き込む
    ///
    /// `init_message_buffer`で事前初期化されたバッファに対して使用する。
    /// base_messageのコピーをスキップし、date/timeのみを書き換える。
    #[inline(always)]
    pub fn write_datetime_only(&self, date_code: u32, time_code: u32, dest: &mut [u32]) {
        debug_assert!(dest.len() >= 16, "destination buffer must be at least 16 elements");
        dest[8] = date_code;
        dest[9] = time_code;
    }
}

// =============================================================================
// テスト
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search_common::params::HardwareType;

    #[test]
    fn test_base_message_builder_creation() {
        let mac = [0x00, 0x09, 0xBF, 0xAA, 0xBB, 0xCC];
        let nazo = [0x02215F10, 0x02215F30, 0x02215F20, 0x02761008, 0x00000000];
        let ds_config = DSConfig::new(mac, nazo, HardwareType::DS);
        let segment = SegmentParams::new(0x1000, 0x60, 0x2FFF);

        let builder = BaseMessageBuilder::from_params(&ds_config, &segment);
        let base = builder.base_message();

        // Verify basic structure
        assert_eq!(base[13], 0x80000000);
        assert_eq!(base[14], 0x00000000);
        assert_eq!(base[15], 0x000001A0);
    }

    #[test]
    fn test_base_message_builder_invalid_mac() {
        let mac = [0x00, 0x09, 0xBF, 0xAA];
        let nazo = [0x02215F10, 0x02215F30, 0x02215F20, 0x02761008, 0x00000000];
        assert!(BaseMessageBuilder::new(&mac, &nazo, 8, 0x1000, 0x60, 0x2FFF).is_err());
    }

    #[test]
    fn test_base_message_builder_invalid_nazo() {
        let mac = [0x00, 0x09, 0xBF, 0xAA, 0xBB, 0xCC];
        let nazo = [0x02215F10, 0x02215F30];
        assert!(BaseMessageBuilder::new(&mac, &nazo, 8, 0x1000, 0x60, 0x2FFF).is_err());
    }

    #[test]
    fn test_build_message() {
        let mac = [0x00, 0x09, 0xBF, 0xAA, 0xBB, 0xCC];
        let nazo = [0x02215F10, 0x02215F30, 0x02215F20, 0x02761008, 0x00000000];
        let ds_config = DSConfig::new(mac, nazo, HardwareType::DS);
        let segment = SegmentParams::new(0x1000, 0x60, 0x2FFF);
        let builder = BaseMessageBuilder::from_params(&ds_config, &segment);

        let message = builder.build_message(0x12345678, 0xABCDEF00);
        assert_eq!(message[8], 0x12345678);
        assert_eq!(message[9], 0xABCDEF00);
    }
}

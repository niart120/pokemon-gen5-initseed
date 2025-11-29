//! ハッシュ値列挙モジュール
//!
//! SHA-1ハッシュ値の列挙処理を提供する。
//! BaseMessageBuilder と DateTimeCodeEnumerator を所有し、
//! ハッシュ値を効率的に列挙する（段階的所有権移譲パターン）。

use super::datetime::{DateTimeCode, DateTimeCodeEnumerator, RangedTimeCodeTable};
use super::message::{BaseMessageBuilder, HashValues};
use crate::sha1::calculate_pokemon_sha1;
use crate::sha1_simd::calculate_pokemon_sha1_simd;

// =============================================================================
// ハッシュ値列挙結果
// =============================================================================

/// ハッシュ値列挙結果（HashValues と DateTimeCode のペア）
#[derive(Debug, Clone, Copy)]
pub struct HashEntry {
    pub hash: HashValues,
    pub datetime_code: DateTimeCode,
}

impl HashEntry {
    pub fn new(hash: HashValues, datetime_code: DateTimeCode) -> Self {
        Self { hash, datetime_code }
    }
}

// =============================================================================
// ハッシュ値列挙器
// =============================================================================

/// ハッシュ値列挙器
///
/// BaseMessageBuilder と DateTimeCodeEnumerator を所有し、
/// SHA-1ハッシュ値を順次生成する。
/// SIMD最適化により4つずつバッチ処理を行う。
pub struct HashValuesEnumerator {
    // 所有権を持つデータ（段階的所有権移譲）
    base_message_builder: BaseMessageBuilder,
    datetime_enumerator: DateTimeCodeEnumerator,

    // SIMD バッファ（Option排除版）
    message_buffer: [u32; 64], // 4メッセージ × 16ワード
    datetime_buffer: [DateTimeCode; 4],
    buffer_len: u8,

    // 出力バッファ（Option排除版）
    output_buffer: [HashEntry; 4],
    output_len: u8,
    output_index: u8,
}

/// ダミーのDateTimeCode（初期化用）
const DUMMY_DATETIME_CODE: DateTimeCode = DateTimeCode {
    date_code: 0,
    time_code: 0,
};

/// ダミーのHashEntry（初期化用）
const DUMMY_HASH_ENTRY: HashEntry = HashEntry {
    hash: HashValues {
        h0: 0,
        h1: 0,
        h2: 0,
        h3: 0,
        h4: 0,
    },
    datetime_code: DUMMY_DATETIME_CODE,
};

impl HashValuesEnumerator {
    /// 新規作成（所有権を受け取る）
    pub fn new(
        base_message_builder: BaseMessageBuilder,
        time_code_table: RangedTimeCodeTable,
        start_seconds: i64,
        range_seconds: u32,
    ) -> Self {
        Self {
            base_message_builder,
            datetime_enumerator: DateTimeCodeEnumerator::new(
                time_code_table,
                start_seconds,
                range_seconds,
            ),
            message_buffer: [0u32; 64],
            datetime_buffer: [DUMMY_DATETIME_CODE; 4],
            buffer_len: 0,
            output_buffer: [DUMMY_HASH_ENTRY; 4],
            output_len: 0,
            output_index: 0,
        }
    }

    /// 処理済み秒数を取得（スキップ分含む）
    pub fn processed_seconds(&self) -> u32 {
        self.datetime_enumerator.processed_seconds()
    }

    /// SIMDバッチ処理を実行（Option排除版）
    fn process_simd_batch(&mut self) {
        if self.buffer_len == 0 {
            return;
        }

        if self.buffer_len == 4 {
            // 4つ揃っている場合はSIMD処理
            let hashes = calculate_pokemon_sha1_simd(&self.message_buffer);

            for i in 0..4 {
                let hash = HashValues::new(
                    hashes[i * 5],
                    hashes[i * 5 + 1],
                    hashes[i * 5 + 2],
                    hashes[i * 5 + 3],
                    hashes[i * 5 + 4],
                );
                self.output_buffer[i] = HashEntry::new(hash, self.datetime_buffer[i]);
            }
            self.output_len = 4;
        } else {
            // 端数はスカラー処理
            let len = self.buffer_len as usize;
            for i in 0..len {
                let offset = i * 16;
                let mut message = [0u32; 16];
                message.copy_from_slice(&self.message_buffer[offset..offset + 16]);

                let (h0, h1, h2, h3, h4) = calculate_pokemon_sha1(&message);
                let hash = HashValues::new(h0, h1, h2, h3, h4);
                self.output_buffer[i] = HashEntry::new(hash, self.datetime_buffer[i]);
            }
            self.output_len = self.buffer_len;
        }

        self.output_index = 0;
        self.buffer_len = 0;
    }

    /// 次のDateTimeCodeをバッファに追加
    fn buffer_next(&mut self) -> bool {
        if let Some(datetime_code) = self.datetime_enumerator.next() {
            let message = self
                .base_message_builder
                .build_message(datetime_code.date_code, datetime_code.time_code);

            let idx = self.buffer_len as usize;
            let offset = idx * 16;
            self.message_buffer[offset..offset + 16].copy_from_slice(&message);
            self.datetime_buffer[idx] = datetime_code;
            self.buffer_len += 1;
            true
        } else {
            false
        }
    }
}

impl Iterator for HashValuesEnumerator {
    type Item = HashEntry;

    fn next(&mut self) -> Option<Self::Item> {
        // 出力バッファに残りがあれば返す
        if self.output_index < self.output_len {
            let entry = self.output_buffer[self.output_index as usize];
            self.output_index += 1;
            return Some(entry);
        }

        // 新しいバッチを収集
        self.buffer_len = 0;
        while self.buffer_len < 4 {
            if !self.buffer_next() {
                break;
            }
        }

        if self.buffer_len == 0 {
            return None;
        }

        // バッチ処理実行
        self.process_simd_batch();

        // 最初の結果を返す
        if self.output_len > 0 {
            let entry = self.output_buffer[0];
            self.output_index = 1;
            return Some(entry);
        }

        None
    }
}

// =============================================================================
// テスト
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search_common::datetime::build_ranged_time_code_table;
    use crate::search_common::params::{HardwareType, DSConfig, SegmentParams, TimeRangeParams};

    fn create_test_ds_config() -> DSConfig {
        let mac = [0x00, 0x09, 0xBF, 0xAA, 0xBB, 0xCC];
        let nazo = [0x02215F10, 0x02215F30, 0x02215F20, 0x02761008, 0x00000000];
        DSConfig::new(mac, nazo, HardwareType::DS)
    }

    fn create_test_segment() -> SegmentParams {
        SegmentParams::new(0x1000, 0x60, 0x2FFF)
    }

    fn create_test_builder() -> BaseMessageBuilder {
        let ds_config = create_test_ds_config();
        let segment = create_test_segment();
        BaseMessageBuilder::from_params(&ds_config, &segment)
    }

    #[test]
    fn test_hash_values_enumerator_basic() {
        let builder = create_test_builder();
        let time_range = TimeRangeParams::new(0, 0, 0, 0, 0, 2).unwrap();
        let table = build_ranged_time_code_table(&time_range, HardwareType::DS);

        let enumerator = HashValuesEnumerator::new(builder, table, 0, 3);
        let results: Vec<HashEntry> = enumerator.collect();

        assert_eq!(results.len(), 3);

        // 各結果がHashValuesとDateTimeCodeを持つことを確認
        for entry in &results {
            // BCDデコードで表示日時が取得できることを確認
            let display = entry.datetime_code.to_display_datetime();
            assert!(display.year >= 2000);
        }
    }

    #[test]
    fn test_hash_values_enumerator_simd_batch() {
        let builder = create_test_builder();

        // 全時間許可で5秒分（SIMD 4 + スカラー 1）
        let time_range = TimeRangeParams::new(0, 23, 0, 59, 0, 59).unwrap();
        let table = build_ranged_time_code_table(&time_range, HardwareType::DS);

        let enumerator = HashValuesEnumerator::new(builder, table, 0, 5);
        let results: Vec<HashEntry> = enumerator.collect();

        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_hash_values_enumerator_consistency() {
        // 同じ入力で同じ結果が得られることを確認
        let builder1 = create_test_builder();
        let time_range1 = TimeRangeParams::new(0, 0, 0, 0, 0, 0).unwrap();
        let table1 = build_ranged_time_code_table(&time_range1, HardwareType::DS);
        let enumerator1 = HashValuesEnumerator::new(builder1, table1, 0, 1);
        let results1: Vec<HashEntry> = enumerator1.collect();

        let builder2 = create_test_builder();
        let time_range2 = TimeRangeParams::new(0, 0, 0, 0, 0, 0).unwrap();
        let table2 = build_ranged_time_code_table(&time_range2, HardwareType::DS);
        let enumerator2 = HashValuesEnumerator::new(builder2, table2, 0, 1);
        let results2: Vec<HashEntry> = enumerator2.collect();

        assert_eq!(results1.len(), results2.len());
        assert_eq!(results1[0].hash.h0, results2[0].hash.h0);
        assert_eq!(results1[0].hash.h1, results2[0].hash.h1);
    }

    #[test]
    fn test_hash_entry_mt_seed() {
        let builder = create_test_builder();
        let time_range = TimeRangeParams::new(0, 0, 0, 0, 0, 0).unwrap();
        let table = build_ranged_time_code_table(&time_range, HardwareType::DS);

        let mut enumerator = HashValuesEnumerator::new(builder, table, 0, 1);
        let entry = enumerator.next().unwrap();

        // MT Seed が計算できることを確認
        let mt_seed = entry.hash.to_mt_seed();
        // u32は常に0以上なので、値が取得できていることを確認
        let _seed = mt_seed; // 値が取得できることを確認
    }
}

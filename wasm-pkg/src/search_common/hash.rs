//! ハッシュ値列挙モジュール
//!
//! SHA-1ハッシュ値の列挙処理を提供する。
//! BaseMessageBuilder と DateTimeCodeEnumerator を所有し、
//! ハッシュ値を効率的に列挙する（段階的所有権移譲パターン）。

use super::datetime::{DateTimeCode, DateTimeCodeEnumerator, RangedTimeCodeTable};
use super::message::BaseMessageBuilder;
use crate::sha1::HashValues;
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
    #[inline]
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
/// SHA-1ハッシュ値を4件ずつ生成する。
/// SIMD最適化により効率的なバッチ処理を行う。
pub struct HashValuesEnumerator {
    base_message_builder: BaseMessageBuilder,
    datetime_enumerator: DateTimeCodeEnumerator,
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
        }
    }

    /// 処理済み秒数を取得（スキップ分含む）
    #[inline]
    pub fn processed_seconds(&self) -> u32 {
        self.datetime_enumerator.processed_seconds()
    }

    /// 次の4件を取得（SIMD処理向け）
    ///
    /// 戻り値: (entries, len)
    /// - entries: 4要素の配列（len未満のインデックスは無効値）
    /// - len: 有効なエントリ数 (0-4)
    ///
    /// len == 0 の場合、列挙終了を意味する
    pub fn next_quad(&mut self) -> ([HashEntry; 4], u8) {
        let (datetime_entries, len) = self.datetime_enumerator.next_quad();
        if len == 0 {
            return ([DUMMY_HASH_ENTRY; 4], 0);
        }

        // バッファをbase_messageで初期化（全スロット）
        let mut message_buffer = [0u32; 64];
        self.base_message_builder.init_message_buffer(&mut message_buffer);

        // date/timeのみを書き込み（base_messageのコピーをスキップ）
        for i in 0..len as usize {
            let offset = i * 16;
            self.base_message_builder.write_datetime_only(
                datetime_entries[i].date_code,
                datetime_entries[i].time_code,
                &mut message_buffer[offset..offset + 16],
            );
        }

        let mut entries = [DUMMY_HASH_ENTRY; 4];

        // 常にSIMD処理（端数でも4件分計算し、必要な分だけ使用）
        let hashes = calculate_pokemon_sha1_simd(&message_buffer);
        for i in 0..len as usize {
            entries[i] = HashEntry::new(hashes[i], datetime_entries[i]);
        }

        (entries, len)
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

    /// next_quad で全件取得するヘルパー
    fn collect_all(enumerator: &mut HashValuesEnumerator) -> Vec<HashEntry> {
        let mut results = Vec::new();
        loop {
            let (entries, len) = enumerator.next_quad();
            if len == 0 {
                break;
            }
            for i in 0..len as usize {
                results.push(entries[i]);
            }
        }
        results
    }

    #[test]
    fn test_hash_values_enumerator_basic() {
        let builder = create_test_builder();
        let time_range = TimeRangeParams::new(0, 0, 0, 0, 0, 2).unwrap();
        let table = build_ranged_time_code_table(&time_range, HardwareType::DS);

        let mut enumerator = HashValuesEnumerator::new(builder, table, 0, 3);
        let results = collect_all(&mut enumerator);

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

        let mut enumerator = HashValuesEnumerator::new(builder, table, 0, 5);
        let results = collect_all(&mut enumerator);

        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_hash_values_enumerator_consistency() {
        // 同じ入力で同じ結果が得られることを確認
        let builder1 = create_test_builder();
        let time_range1 = TimeRangeParams::new(0, 0, 0, 0, 0, 0).unwrap();
        let table1 = build_ranged_time_code_table(&time_range1, HardwareType::DS);
        let mut enumerator1 = HashValuesEnumerator::new(builder1, table1, 0, 1);
        let results1 = collect_all(&mut enumerator1);

        let builder2 = create_test_builder();
        let time_range2 = TimeRangeParams::new(0, 0, 0, 0, 0, 0).unwrap();
        let table2 = build_ranged_time_code_table(&time_range2, HardwareType::DS);
        let mut enumerator2 = HashValuesEnumerator::new(builder2, table2, 0, 1);
        let results2 = collect_all(&mut enumerator2);

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
        let (entries, len) = enumerator.next_quad();
        assert_eq!(len, 1);

        // MT Seed が計算できることを確認
        let mt_seed = entries[0].hash.to_mt_seed();
        let _seed = mt_seed; // 値が取得できることを確認
    }

    #[test]
    fn test_next_quad_basic() {
        let builder = create_test_builder();
        let time_range = TimeRangeParams::new(0, 23, 0, 59, 0, 59).unwrap();
        let table = build_ranged_time_code_table(&time_range, HardwareType::DS);

        let mut enumerator = HashValuesEnumerator::new(builder, table, 0, 10);

        // 最初の4件
        let (entries1, len1) = enumerator.next_quad();
        assert_eq!(len1, 4);

        // 次の4件
        let (entries2, len2) = enumerator.next_quad();
        assert_eq!(len2, 4);

        // 残り2件
        let (entries3, len3) = enumerator.next_quad();
        assert_eq!(len3, 2);

        // 終了
        let (_, len4) = enumerator.next_quad();
        assert_eq!(len4, 0);

        // エントリが異なることを確認
        assert_ne!(entries1[0].hash.h0, entries2[0].hash.h0);
        assert_ne!(entries2[0].hash.h0, entries3[0].hash.h0);
    }
}

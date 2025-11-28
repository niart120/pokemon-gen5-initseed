//! 日時コード生成モジュール
//!
//! 日時コード計算・列挙とユーティリティ関数を提供する。

use super::SECONDS_PER_DAY;
use crate::datetime_codes::{DateCodeGenerator, TimeCodeGenerator};
use crate::search_common::params::{HardwareType, TimeRangeParams};
use crate::utils::NumberUtils;

// =============================================================================
// 日時コード
// =============================================================================

/// 表示用日時情報
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayDateTime {
    pub year: u32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
}

/// 日時コード（date_code と time_code のペア）
///
/// - `date_code`: 0xYYMMDDWW (BCD形式、WW=曜日)
/// - `time_code`: 0xHHMMSS00 (BCD形式、下位8bitは未使用)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DateTimeCode {
    pub date_code: u32,
    pub time_code: u32,
}

impl DateTimeCode {
    pub fn new(date_code: u32, time_code: u32) -> Self {
        Self { date_code, time_code }
    }

    /// 表示用日時に変換（BCDデコード）
    pub fn to_display_datetime(&self) -> DisplayDateTime {
        // date_code: 0xYYMMDDWW
        let yy = NumberUtils::decode_bcd((self.date_code >> 24) as u8) as u32;
        let mm = NumberUtils::decode_bcd((self.date_code >> 16) as u8) as u32;
        let dd = NumberUtils::decode_bcd((self.date_code >> 8) as u8) as u32;
        // WW (曜日) は下位8bitだが、表示には不要

        // time_code: 0xHHMMSS00
        // 注: HHは24時間制でエンコードされている
        // DS/DS_LITEでは0x40000000フラグが午後に付くが、HH自体は24時間制のまま
        let hh = NumberUtils::decode_bcd((self.time_code >> 24) as u8) as u32;
        let mi = NumberUtils::decode_bcd((self.time_code >> 16) as u8) as u32;
        let ss = NumberUtils::decode_bcd((self.time_code >> 8) as u8) as u32;

        DisplayDateTime {
            year: 2000 + yy,
            month: mm,
            day: dd,
            hour: hh,
            minute: mi,
            second: ss,
        }
    }
}

// =============================================================================
// 範囲制限タイムコードテーブル
// =============================================================================

/// 範囲制限タイムコードテーブル
///
/// 86,400要素の配列で、各インデックスが1日の秒数（0-86399）に対応する。
/// `Some(time_code)` なら検索対象、`None` なら対象外。
pub type RangedTimeCodeTable = Box<[Option<u32>; 86400]>;

/// 範囲制限タイムコードテーブルを構築
///
/// 許可秒に対応する time_code を事前計算し、O(1)でアクセス可能にする。
pub fn build_ranged_time_code_table(range: &TimeRangeParams, hardware: HardwareType) -> RangedTimeCodeTable {
    let mut table: RangedTimeCodeTable = Box::new([None; 86400]);
    let hardware_str = hardware.as_str();

    for hour in range.hour_start..=range.hour_end {
        for minute in range.minute_start..=range.minute_end {
            for second in range.second_start..=range.second_end {
                let second_of_day = hour * 3600 + minute * 60 + second;
                let time_code =
                    TimeCodeGenerator::get_time_code_for_hardware(second_of_day, hardware_str);
                table[second_of_day as usize] = Some(time_code);
            }
        }
    }
    table
}

// =============================================================================
// 日時コード列挙器
// =============================================================================

/// 日時コードバッチ（固定長配列 + 長さカウンタ）
///
/// Option を使用しない設計でパフォーマンスを最適化。
#[derive(Debug)]
pub struct DateTimeBatch {
    entries: [DateTimeCode; 4],
    len: u8,
}

/// ダミーのDateTimeCode（初期化用）
const DUMMY_DATETIME_CODE: DateTimeCode = DateTimeCode {
    date_code: 0,
    time_code: 0,
};

impl DateTimeBatch {
    /// バッチ内の有効エントリ数を取得
    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// バッチが空かどうかを確認
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 有効エントリのスライスを取得
    #[inline]
    pub fn entries(&self) -> &[DateTimeCode] {
        &self.entries[..self.len as usize]
    }
}

/// 日時コード列挙器
///
/// RangedTimeCodeTable を所有し、開始時刻から指定秒数分の DateTimeCode を
/// 順次生成する Iterator。
/// 許可範囲外の秒はスキップされるが、進捗計算にはスキップ分も含まれる。
pub struct DateTimeCodeEnumerator {
    time_code_table: RangedTimeCodeTable,
    current_seconds: i64,
    end_seconds: i64,
    processed_seconds: u32,
}

impl DateTimeCodeEnumerator {
    /// 新規作成（所有権を受け取る）
    ///
    /// # Arguments
    /// - `time_code_table`: 範囲制限タイムコードテーブル（許可秒のtime_codeを含む）
    /// - `start_seconds`: 開始秒（2000年からの経過秒）
    /// - `range_seconds`: 検索範囲（秒数）
    pub fn new(
        time_code_table: RangedTimeCodeTable,
        start_seconds: i64,
        range_seconds: u32,
    ) -> Self {
        Self {
            time_code_table,
            current_seconds: start_seconds,
            end_seconds: start_seconds + range_seconds as i64,
            processed_seconds: 0,
        }
    }

    /// 処理済み秒数を取得（スキップ分含む）
    pub fn processed_seconds(&self) -> u32 {
        self.processed_seconds
    }

    /// バッチイテレータを取得（for文での利用向け）
    ///
    /// `for batch in enumerator.batches() { ... }` の形式で使用可能。
    /// 各バッチは最大4エントリを含む。
    pub fn batches(self) -> DateTimeBatchIterator {
        DateTimeBatchIterator { inner: self }
    }

    /// 次のバッチを取得（内部用）
    fn next_batch(&mut self) -> Option<DateTimeBatch> {
        let mut batch = DateTimeBatch {
            entries: [DUMMY_DATETIME_CODE; 4],
            len: 0,
        };

        while batch.len < 4 {
            if let Some(datetime_code) = self.next() {
                batch.entries[batch.len as usize] = datetime_code;
                batch.len += 1;
            } else {
                break;
            }
        }

        if batch.len == 0 {
            None
        } else {
            Some(batch)
        }
    }
}

/// バッチイテレータ（for文での利用向け）
pub struct DateTimeBatchIterator {
    inner: DateTimeCodeEnumerator,
}

impl Iterator for DateTimeBatchIterator {
    type Item = DateTimeBatch;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next_batch()
    }
}

impl Iterator for DateTimeCodeEnumerator {
    type Item = DateTimeCode;

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_seconds < self.end_seconds {
            let seconds = self.current_seconds;
            self.current_seconds += 1;
            self.processed_seconds += 1;

            if seconds < 0 {
                continue;
            }

            let second_of_day = (seconds % SECONDS_PER_DAY) as usize;
            if let Some(time_code) = self.time_code_table[second_of_day] {
                let date_index = (seconds / SECONDS_PER_DAY) as u32;
                let date_code = DateCodeGenerator::get_date_code(date_index);
                return Some(DateTimeCode::new(date_code, time_code));
            }
        }
        None
    }
}



// =============================================================================
// テスト
// =============================================================================

#[cfg(test)]
mod tests {
    use crate::search_common::EPOCH_2000_UNIX;
    use super::*;
    
    // =============================================================================
    // ユーティリティ関数
    // =============================================================================

    
    use chrono::{Datelike, NaiveDate, Timelike};

    /// 結果表示用の日時を生成
    ///
    /// seconds_since_2000 から (year, month, day, hour, minute, second) を生成する。
    fn generate_display_datetime(
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
    fn datetime_to_seconds_since_2000(
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
    fn date_to_seconds_since_2000(year: u32, month: u32, day: u32) -> Option<i64> {
        datetime_to_seconds_since_2000(year, month, day, 0, 0, 0)
    }

    fn create_test_time_range() -> TimeRangeParams {
        TimeRangeParams::new(0, 0, 0, 0, 0, 2).unwrap()
    }

    #[test]
    fn test_date_time_code_creation() {
        let dtc = DateTimeCode::new(0x12345678, 0xABCDEF00);
        assert_eq!(dtc.date_code, 0x12345678);
        assert_eq!(dtc.time_code, 0xABCDEF00);
    }

    #[test]
    fn test_date_time_code_to_display_datetime() {
        // 2000年1月1日 0:00:00 (date_code: 0x00010106 = YY=00, MM=01, DD=01, WW=06(土))
        // time_code: 0x00000000 = HH=00, MM=00, SS=00
        let dtc = DateTimeCode::new(0x00010106, 0x00000000);
        let display = dtc.to_display_datetime();
        assert_eq!(display.year, 2000);
        assert_eq!(display.month, 1);
        assert_eq!(display.day, 1);
        assert_eq!(display.hour, 0);
        assert_eq!(display.minute, 0);
        assert_eq!(display.second, 0);

        // 2024年6月15日 12:30:45
        // date_code: 0x24061505 = YY=24, MM=06, DD=15, WW=05(金)
        // time_code: 0x12304500 = HH=12, MM=30, SS=45 (24時間制)
        let dtc = DateTimeCode::new(0x24061505, 0x12304500);
        let display = dtc.to_display_datetime();
        assert_eq!(display.year, 2024);
        assert_eq!(display.month, 6);
        assert_eq!(display.day, 15);
        assert_eq!(display.hour, 12);
        assert_eq!(display.minute, 30);
        assert_eq!(display.second, 45);
    }

    #[test]
    fn test_bcd_decode() {
        // BCD デコードの動作確認（NumberUtils::decode_bcd を使用）
        assert_eq!(NumberUtils::decode_bcd(0x00), 0);
        assert_eq!(NumberUtils::decode_bcd(0x09), 9);
        assert_eq!(NumberUtils::decode_bcd(0x10), 10);
        assert_eq!(NumberUtils::decode_bcd(0x23), 23);
        assert_eq!(NumberUtils::decode_bcd(0x59), 59);
        assert_eq!(NumberUtils::decode_bcd(0x99), 99);
    }

    #[test]
    fn test_build_ranged_time_code_table() {
        let range = create_test_time_range();
        let table = build_ranged_time_code_table(&range, HardwareType::DS);

        // 0, 1, 2秒目は許可されている
        assert!(table[0].is_some());
        assert!(table[1].is_some());
        assert!(table[2].is_some());
        // 3秒目以降は許可されていない
        assert!(table[3].is_none());
        assert!(table[3600].is_none());
    }

    #[test]
    fn test_ranged_time_code_table_values() {
        let range = TimeRangeParams::new(0, 0, 0, 0, 0, 0).unwrap();
        let table = build_ranged_time_code_table(&range, HardwareType::DS);

        let expected_time_code = TimeCodeGenerator::get_time_code_for_hardware(0, "DS");
        assert_eq!(table[0], Some(expected_time_code));
    }

    #[test]
    fn test_datetime_enumerator_basic() {
        let range = create_test_time_range();
        let table = build_ranged_time_code_table(&range, HardwareType::DS);

        // 2000年1月1日 0:00:00 から開始
        let enumerator = DateTimeCodeEnumerator::new(table, 0, 3);

        let results: Vec<DateTimeCode> = enumerator.collect();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_datetime_enumerator_skips_disallowed() {
        // 1秒目のみ許可
        let range = TimeRangeParams::new(0, 0, 0, 0, 1, 1).unwrap();
        let table = build_ranged_time_code_table(&range, HardwareType::DS);

        // 0秒目から5秒間
        let enumerator = DateTimeCodeEnumerator::new(table, 0, 5);

        let results: Vec<DateTimeCode> = enumerator.collect();
        // 0, 2, 3, 4秒目はスキップされ、1秒目のみ返される
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_datetime_enumerator_processed_seconds() {
        let range = TimeRangeParams::new(0, 0, 0, 0, 1, 1).unwrap();
        let table = build_ranged_time_code_table(&range, HardwareType::DS);

        let mut enumerator = DateTimeCodeEnumerator::new(table, 0, 5);

        // 全て消費
        while enumerator.next().is_some() {}

        // スキップ分も含めて5秒処理された
        assert_eq!(enumerator.processed_seconds(), 5);
    }

    #[test]
    fn test_datetime_enumerator_across_days() {
        // 全時間許可
        let range = TimeRangeParams::new(0, 23, 0, 59, 0, 59).unwrap();
        let table = build_ranged_time_code_table(&range, HardwareType::DS);

        // 1日目の最後から2日目の最初にまたがる
        let start = SECONDS_PER_DAY - 2; // 86398秒目
        let enumerator = DateTimeCodeEnumerator::new(table, start, 4);

        let results: Vec<DateTimeCode> = enumerator.collect();
        assert_eq!(results.len(), 4);

        // date_codeが変わることを確認
        let date_code_day1 = DateCodeGenerator::get_date_code(0);
        let date_code_day2 = DateCodeGenerator::get_date_code(1);

        // 最初の2つは1日目、残りは2日目
        assert_eq!(results[0].date_code, date_code_day1);
        assert_eq!(results[1].date_code, date_code_day1);
        assert_eq!(results[2].date_code, date_code_day2);
        assert_eq!(results[3].date_code, date_code_day2);
    }

    #[test]
    fn test_datetime_to_seconds_since_2000() {
        // 2000年1月1日 0:00:00
        let result = datetime_to_seconds_since_2000(2000, 1, 1, 0, 0, 0);
        assert_eq!(result, Some(0));

        // 2000年1月2日 0:00:00
        let result = datetime_to_seconds_since_2000(2000, 1, 2, 0, 0, 0);
        assert_eq!(result, Some(86400));
    }

    #[test]
    fn test_date_to_seconds_since_2000() {
        let result = date_to_seconds_since_2000(2000, 1, 1);
        assert_eq!(result, Some(0));

        let result = date_to_seconds_since_2000(2024, 1, 15);
        assert!(result.unwrap() > 0);
    }

    #[test]
    fn test_date_to_seconds_since_2000_invalid() {
        // 無効な日付
        assert!(date_to_seconds_since_2000(2024, 13, 1).is_none());
        assert!(date_to_seconds_since_2000(2024, 2, 30).is_none());
    }

    #[test]
    fn test_generate_display_datetime() {
        // 2000年1月1日 0:00:00
        let result = generate_display_datetime(0);
        assert_eq!(result, Some((2000, 1, 1, 0, 0, 0)));

        // 2000年1月1日 1:00:00
        let result = generate_display_datetime(3600);
        assert_eq!(result, Some((2000, 1, 1, 1, 0, 0)));

        // 2000年1月2日 0:00:00
        let result = generate_display_datetime(86400);
        assert_eq!(result, Some((2000, 1, 2, 0, 0, 0)));
    }

    #[test]
    fn test_generate_display_datetime_2024() {
        // 2024年のある時点
        let seconds = datetime_to_seconds_since_2000(2024, 6, 15, 12, 30, 45).unwrap();
        let result = generate_display_datetime(seconds);
        assert_eq!(result, Some((2024, 6, 15, 12, 30, 45)));
    }

    #[test]
    fn test_batch_iterator() {
        // 全時間許可
        let range = TimeRangeParams::new(0, 23, 0, 59, 0, 59).unwrap();
        let table = build_ranged_time_code_table(&range, HardwareType::DS);

        let enumerator = DateTimeCodeEnumerator::new(table, 0, 10);
        
        let mut total_entries = 0;
        for batch in enumerator.batches() {
            assert!(!batch.is_empty());
            assert!(batch.len() <= 4);
            for _entry in batch.entries() {
                total_entries += 1;
            }
        }
        
        assert_eq!(total_entries, 10);
    }
}

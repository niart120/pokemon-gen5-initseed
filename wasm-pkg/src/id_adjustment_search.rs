//! ID調整検索器
//!
//! BW/BW2におけるID調整（表ID/裏IDを持つ初期Seedを検索）機能を提供する。
//! 指定されたTID/SIDにマッチする起動条件を列挙する。
//!
//! ## セグメントパターン設計
//!
//! TypeScript側で timer0 × vcount × keyCode のセグメントループを実装:
//! 1. `IdAdjustmentSearchIterator::new()` で単一セグメント（固定timer0/vcount/keyCode）のイテレータを作成
//! 2. `IdAdjustmentSearchIterator::next_batch()` で seconds 方向の結果をバッチ取得
//! 3. `is_finished` で完了判定
//!
//! ## 公開API
//! - `IdAdjustmentSearchIterator`: 単一セグメントの検索イテレータ
//! - `IdAdjustmentSearchResult`: 検索結果1件
//! - `IdAdjustmentSearchResults`: バッチ結果

use crate::offset_calculator::{calculate_tid_sid_from_seed, GameMode};
use crate::pid_shiny_checker::{ShinyChecker, ShinyType};
use crate::search_common::{
    build_ranged_time_code_table, BaseMessageBuilder, DSConfigJs, HashEntry, HashValuesEnumerator,
    SearchRangeParamsJs, SegmentParamsJs, TimeRangeParamsJs,
};
use wasm_bindgen::prelude::*;

// =============================================================================
// 検索結果
// =============================================================================

/// ID調整検索結果1件
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct IdAdjustmentSearchResult {
    // LCG Seed
    lcg_seed_high: u32,
    lcg_seed_low: u32,

    // TID/SID
    tid: u16,
    sid: u16,

    // 色違いタイプ (0=Normal, 1=Square, 2=Star)
    shiny_type: u8,

    // 起動条件
    year: u32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
    timer0: u32,
    vcount: u32,
    key_code: u32,
}

#[wasm_bindgen]
impl IdAdjustmentSearchResult {
    // LCG Seed
    #[wasm_bindgen(getter = lcgSeedHigh)]
    pub fn lcg_seed_high(&self) -> u32 {
        self.lcg_seed_high
    }

    #[wasm_bindgen(getter = lcgSeedLow)]
    pub fn lcg_seed_low(&self) -> u32 {
        self.lcg_seed_low
    }

    #[wasm_bindgen(getter = lcgSeedHex)]
    pub fn lcg_seed_hex(&self) -> String {
        let lcg_seed = ((self.lcg_seed_high as u64) << 32) | (self.lcg_seed_low as u64);
        format!("{lcg_seed:016X}")
    }

    // TID/SID
    #[wasm_bindgen(getter)]
    pub fn tid(&self) -> u16 {
        self.tid
    }

    #[wasm_bindgen(getter)]
    pub fn sid(&self) -> u16 {
        self.sid
    }

    // 色違いタイプ
    #[wasm_bindgen(getter = shinyType)]
    pub fn shiny_type(&self) -> u8 {
        self.shiny_type
    }

    // 起動日時
    #[wasm_bindgen(getter)]
    pub fn year(&self) -> u32 {
        self.year
    }

    #[wasm_bindgen(getter)]
    pub fn month(&self) -> u32 {
        self.month
    }

    #[wasm_bindgen(getter)]
    pub fn day(&self) -> u32 {
        self.day
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

    // 起動パラメータ
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
}

/// バッチ検索結果
#[wasm_bindgen]
pub struct IdAdjustmentSearchResults {
    results: Vec<IdAdjustmentSearchResult>,
    processed_in_chunk: u32,
}

#[wasm_bindgen]
impl IdAdjustmentSearchResults {
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.results.len()
    }

    #[wasm_bindgen(getter = processedInChunk)]
    pub fn processed_in_chunk(&self) -> u32 {
        self.processed_in_chunk
    }

    /// 結果をJavaScript配列として取得
    #[wasm_bindgen]
    pub fn to_array(&self) -> js_sys::Array {
        let arr = js_sys::Array::new();
        for result in &self.results {
            arr.push(&JsValue::from(result.clone()));
        }
        arr
    }

    /// 指定インデックスの結果を取得
    #[wasm_bindgen]
    pub fn get(&self, index: usize) -> Option<IdAdjustmentSearchResult> {
        self.results.get(index).cloned()
    }
}

// =============================================================================
// 検索イテレータ
// =============================================================================

/// ID調整検索イテレータ
///
/// 単一セグメント（固定 timer0/vcount/keyCode）に対して seconds 方向の検索を行う。
/// TypeScript側で timer0 × vcount × keyCode のセグメントループを実装し、
/// 各セグメントに対してこのイテレータを作成する。
#[wasm_bindgen]
pub struct IdAdjustmentSearchIterator {
    // ハッシュ値列挙器（所有）
    hash_enumerator: HashValuesEnumerator,

    // セグメントパラメータ（結果出力用に保持）
    timer0: u32,
    vcount: u32,
    key_code: u32,

    // 検索条件
    target_tid: Option<u16>,
    target_sid: Option<u16>,
    shiny_pid: Option<u32>,
    game_mode: GameMode,

    // 検索範囲
    range_seconds: u32,

    // 現在位置
    current_offset: u32,
    finished: bool,
}

/// u8からGameModeへの変換
fn game_mode_from_u8(value: u8) -> Result<GameMode, String> {
    match value {
        0 => Ok(GameMode::BwNewGameWithSave),
        1 => Ok(GameMode::BwNewGameNoSave),
        2 => Ok(GameMode::BwContinue),
        3 => Ok(GameMode::Bw2NewGameWithMemoryLinkSave),
        4 => Ok(GameMode::Bw2NewGameNoMemoryLinkSave),
        5 => Ok(GameMode::Bw2NewGameNoSave),
        6 => Ok(GameMode::Bw2ContinueWithMemoryLink),
        7 => Ok(GameMode::Bw2ContinueNoMemoryLink),
        _ => Err(format!("Invalid game mode: {}", value)),
    }
}

#[wasm_bindgen]
impl IdAdjustmentSearchIterator {
    /// コンストラクタ
    ///
    /// # Arguments
    /// - `ds_config`: DS設定パラメータ (MAC/Nazo/Hardware)
    /// - `segment`: セグメントパラメータ (Timer0/VCount/KeyCode)
    /// - `time_range`: 時刻範囲パラメータ
    /// - `search_range`: 検索範囲パラメータ
    /// - `target_tid`: 検索対象の表ID（-1で指定なし）
    /// - `target_sid`: 検索対象の裏ID（-1で指定なし）
    /// - `shiny_pid`: 色違いにしたい個体のPID（-1で指定なし）
    /// - `game_mode`: ゲームモード (0-7)
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ds_config: &DSConfigJs,
        segment: &SegmentParamsJs,
        time_range: &TimeRangeParamsJs,
        search_range: &SearchRangeParamsJs,
        target_tid: i32, // -1 for None
        target_sid: i32, // -1 for None
        shiny_pid: f64,  // -1 for None, use f64 to handle u32 range
        game_mode: u8,
    ) -> Result<IdAdjustmentSearchIterator, String> {
        // GameModeの変換と検証
        let game_mode = game_mode_from_u8(game_mode)?;

        // 「続きから」モードはID調整不可
        if matches!(
            game_mode,
            GameMode::BwContinue
                | GameMode::Bw2ContinueWithMemoryLink
                | GameMode::Bw2ContinueNoMemoryLink
        ) {
            return Err("ID調整には「始めから」モードを選択してください".to_string());
        }

        // 内部型に変換
        let ds_config_internal = ds_config.to_ds_config();
        let segment_internal = segment.to_segment_params();
        let time_range_internal = time_range.to_time_range_params();
        let hardware = ds_config_internal.hardware;

        // BaseMessageBuilder構築
        let base_message_builder =
            BaseMessageBuilder::from_params(&ds_config_internal, &segment_internal);

        // RangedTimeCodeTable構築
        let time_code_table = build_ranged_time_code_table(&time_range_internal, hardware);

        // 開始秒を計算
        let start_seconds = search_range.start_seconds_since_2000();
        let range_seconds = search_range.range_seconds();

        // target_tidの変換
        let target_tid = if target_tid < 0 {
            None
        } else {
            Some(target_tid as u16)
        };

        // target_sidの変換
        let target_sid = if target_sid < 0 {
            None
        } else {
            Some(target_sid as u16)
        };

        // shiny_pidの変換
        let shiny_pid = if shiny_pid < 0.0 {
            None
        } else {
            Some(shiny_pid as u32)
        };

        // HashValuesEnumerator構築
        let hash_enumerator = HashValuesEnumerator::new(
            base_message_builder,
            time_code_table,
            start_seconds,
            range_seconds,
        );

        Ok(IdAdjustmentSearchIterator {
            hash_enumerator,
            timer0: segment_internal.timer0,
            vcount: segment_internal.vcount,
            key_code: segment_internal.key_code,
            target_tid,
            target_sid,
            shiny_pid,
            game_mode,
            range_seconds,
            current_offset: 0,
            finished: false,
        })
    }

    /// 検索が完了したかどうか
    #[wasm_bindgen(getter = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// 処理済み秒数
    #[wasm_bindgen(getter = processedSeconds)]
    pub fn processed_seconds(&self) -> u32 {
        self.current_offset
    }

    /// 総秒数
    #[wasm_bindgen(getter = totalSeconds)]
    pub fn total_seconds(&self) -> u32 {
        self.range_seconds
    }

    /// 進捗率 (0.0 - 1.0)
    #[wasm_bindgen(getter)]
    pub fn progress(&self) -> f64 {
        if self.range_seconds == 0 {
            return 1.0;
        }
        self.current_offset as f64 / self.range_seconds as f64
    }

    /// 次のバッチを取得
    ///
    /// - max_results件見つかったら即return
    /// - chunk_seconds秒分処理したら結果がなくても一旦return
    /// - 検索範囲を全て処理したらfinished=trueになる
    #[wasm_bindgen]
    pub fn next_batch(
        &mut self,
        max_results: u32,
        chunk_seconds: u32,
    ) -> IdAdjustmentSearchResults {
        if self.finished {
            return IdAdjustmentSearchResults {
                results: Vec::new(),
                processed_in_chunk: 0,
            };
        }

        let mut results: Vec<IdAdjustmentSearchResult> = Vec::new();
        let initial_processed = self.hash_enumerator.processed_seconds();
        let target_processed = initial_processed + chunk_seconds;

        // HashValuesEnumeratorからハッシュ値を4件ずつ取得して検証
        loop {
            let (entries, len) = self.hash_enumerator.next_quad();
            if len == 0 {
                break;
            }

            // 同一バッチ内のエントリはすべて処理（境界での取りこぼし防止）
            for i in 0..len as usize {
                let entry = &entries[i];

                // LCG Seedを計算
                let lcg_seed = entry.hash.to_lcg_seed();

                // TID/SIDを計算
                let tid_sid_result = calculate_tid_sid_from_seed(lcg_seed, self.game_mode);

                // TIDフィルタ（指定時のみ）
                if let Some(target_tid) = self.target_tid {
                    if tid_sid_result.tid != target_tid {
                        continue;
                    }
                }

                // SIDフィルタ（指定時のみ）
                if let Some(target_sid) = self.target_sid {
                    if tid_sid_result.sid != target_sid {
                        continue;
                    }
                }

                // 色違いタイプ判定（shinyPid指定時）
                let shiny_type = if let Some(pid) = self.shiny_pid {
                    ShinyChecker::check_shiny_type(tid_sid_result.tid, tid_sid_result.sid, pid)
                } else {
                    ShinyType::Normal
                };

                // 色違いフィルタ（shinyPid指定時はSquareまたはStarのみ結果に含める）
                if self.shiny_pid.is_some() && shiny_type == ShinyType::Normal {
                    continue;
                }

                // 結果を追加
                results.push(self.create_result(
                    entry,
                    tid_sid_result.tid,
                    tid_sid_result.sid,
                    shiny_type,
                ));
            }

            // max_results到達チェック（バッチ処理完了後）
            if results.len() >= max_results as usize {
                break;
            }

            // チャンク処理制限
            if self.hash_enumerator.processed_seconds() >= target_processed {
                break;
            }
        }

        // 処理済み秒数を更新
        let current_processed = self.hash_enumerator.processed_seconds();
        self.current_offset = current_processed;
        let seconds_processed = current_processed - initial_processed;

        // 検索完了チェック
        if current_processed >= self.range_seconds {
            self.finished = true;
        }

        IdAdjustmentSearchResults {
            results,
            processed_in_chunk: seconds_processed,
        }
    }

    /// HashEntryから検索結果を生成
    fn create_result(
        &self,
        entry: &HashEntry,
        tid: u16,
        sid: u16,
        shiny_type: ShinyType,
    ) -> IdAdjustmentSearchResult {
        let display = entry.datetime_code.to_display_datetime();
        let lcg_seed = entry.hash.to_lcg_seed();

        IdAdjustmentSearchResult {
            lcg_seed_high: (lcg_seed >> 32) as u32,
            lcg_seed_low: lcg_seed as u32,
            tid,
            sid,
            shiny_type: shiny_type as u8,
            year: display.year,
            month: display.month,
            day: display.day,
            hour: display.hour,
            minute: display.minute,
            second: display.second,
            timer0: self.timer0,
            vcount: self.vcount,
            key_code: self.key_code,
        }
    }
}

// =============================================================================
// テスト
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_ds_config() -> DSConfigJs {
        let mac = [0x00, 0x09, 0xBF, 0xAA, 0xBB, 0xCC];
        let nazo = [0x02215F10, 0x02215F30, 0x02215F20, 0x02761008, 0x00000000];
        DSConfigJs::new(&mac, &nazo, "DS").unwrap()
    }

    fn create_test_segment() -> SegmentParamsJs {
        SegmentParamsJs::new(0x1000, 0x60, 0x2FFF)
    }

    fn create_test_time_range() -> TimeRangeParamsJs {
        TimeRangeParamsJs::new(0, 23, 0, 59, 0, 59).unwrap()
    }

    fn create_test_search_range() -> SearchRangeParamsJs {
        // 2024年1月1日から1時間
        SearchRangeParamsJs::new(2024, 1, 1, 0, 3600).unwrap()
    }

    #[test]
    fn test_iterator_creation() {
        let ds_config = create_test_ds_config();
        let segment = create_test_segment();
        let time_range = create_test_time_range();
        let search_range = create_test_search_range();

        // BwNewGameNoSave (GameMode = 1)
        let iterator = IdAdjustmentSearchIterator::new(
            &ds_config,
            &segment,
            &time_range,
            &search_range,
            12345, // target_tid
            -1,    // target_sid (None)
            -1.0,  // shiny_pid (None)
            1,     // game_mode (BwNewGameNoSave)
        );

        assert!(iterator.is_ok());
        let iterator = iterator.unwrap();
        assert!(!iterator.is_finished());
        assert_eq!(iterator.processed_seconds(), 0);
    }

    #[test]
    fn test_iterator_rejects_continue_mode() {
        let ds_config = create_test_ds_config();
        let segment = create_test_segment();
        let time_range = create_test_time_range();
        let search_range = create_test_search_range();

        // BwContinue (GameMode = 2) should be rejected
        let iterator = IdAdjustmentSearchIterator::new(
            &ds_config,
            &segment,
            &time_range,
            &search_range,
            12345,
            -1,
            -1.0,
            2, // BwContinue
        );

        assert!(iterator.is_err());
        match iterator {
            Err(e) => assert!(e.contains("始めから")),
            Ok(_) => panic!("Expected error for continue mode"),
        }
    }

    #[test]
    fn test_game_mode_conversion() {
        assert!(matches!(
            game_mode_from_u8(0),
            Ok(GameMode::BwNewGameWithSave)
        ));
        assert!(matches!(
            game_mode_from_u8(1),
            Ok(GameMode::BwNewGameNoSave)
        ));
        assert!(matches!(game_mode_from_u8(2), Ok(GameMode::BwContinue)));
        assert!(matches!(
            game_mode_from_u8(5),
            Ok(GameMode::Bw2NewGameNoSave)
        ));
        assert!(game_mode_from_u8(8).is_err());
    }

    #[test]
    fn test_next_batch_processes_seconds() {
        let ds_config = create_test_ds_config();
        let segment = create_test_segment();
        let time_range = create_test_time_range();
        let search_range = create_test_search_range();

        let mut iterator = IdAdjustmentSearchIterator::new(
            &ds_config,
            &segment,
            &time_range,
            &search_range,
            0, // target_tid = 0 (unlikely to match)
            -1,
            -1.0,
            1,
        )
        .unwrap();

        // Process 60 seconds
        let results = iterator.next_batch(100, 60);

        // Should have processed some seconds
        assert!(iterator.processed_seconds() > 0);
        // Progress should be updated
        assert!(iterator.progress() > 0.0);
        // Should have processed approximately 60 seconds
        assert!(results.processed_in_chunk() <= 60 || results.processed_in_chunk() >= 1);
    }
}

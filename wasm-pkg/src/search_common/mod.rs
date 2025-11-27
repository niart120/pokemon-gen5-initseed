//! 検索処理共通モジュール
//!
//! 起動時間検索で共通利用される定数・型・ユーティリティ関数を提供する。
//!
//! ## サブモジュール構成
//!
//! - `params`: パラメータ型（内部型 + 公開型）
//! - `message`: SHA-1メッセージ構築
//! - `datetime`: 日時コード列挙・ユーティリティ
//! - `hash`: ハッシュ値列挙器

mod datetime;
mod hash;
mod message;
mod params;

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
// Re-exports
// =============================================================================

// params モジュール（公開型）
pub use params::{
    DSConfigJs, SearchRangeParamsJs, SegmentParamsJs, TimeRangeParamsJs,
};

// params モジュール（内部型 - テスト用にも公開）
pub use params::{
    DSConfig, HardwareType, SearchRangeParams, SegmentParams, TimeRangeParams,
};

// message モジュール
pub use message::BaseMessageBuilder;

// datetime モジュール
pub use datetime::{
    build_ranged_time_code_table, DisplayDateTime,
};

// hash モジュール
pub use hash::{HashEntry, HashValuesEnumerator};

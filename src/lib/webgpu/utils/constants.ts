/**
 * WebGPU共通定数
 */

// バッファアライメント
export const BUFFER_ALIGNMENT = 256;

// メッセージ・ハッシュ関連（SHA-1検索用）
export const WORDS_PER_MESSAGE = 16;
export const WORDS_PER_HASH = 5;
export const BYTES_PER_WORD = Uint32Array.BYTES_PER_ELEMENT;
export const BYTES_PER_MESSAGE = WORDS_PER_MESSAGE * BYTES_PER_WORD;
export const BYTES_PER_HASH = WORDS_PER_HASH * BYTES_PER_WORD;

// マッチ結果レコード
export const MATCH_RECORD_WORDS = 2;
export const MATCH_RECORD_BYTES = MATCH_RECORD_WORDS * BYTES_PER_WORD;
export const MATCH_OUTPUT_HEADER_WORDS = 1;
export const MATCH_OUTPUT_HEADER_BYTES = MATCH_OUTPUT_HEADER_WORDS * BYTES_PER_WORD;

// ダブルバッファリング
export const DOUBLE_BUFFER_SET_COUNT = 8;

// MT Seed検索用
export const MT_SEED_SEARCH_PARAMS_WORDS = 8;
export const MT_SEED_RESULT_HEADER_WORDS = 1;
export const MT_SEED_RESULT_RECORD_WORDS = 2;
export const MT_SEED_MAX_RESULTS_PER_DISPATCH = 4096;

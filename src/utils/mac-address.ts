/**
 * MACアドレス変換ユーティリティ
 *
 * - number[] や string[]("0x12"/"12") を受け付けて Uint8Array(6) に正規化
 * - 値は 0-255 にクランプ、少数は切り捨て
 * - 長さ不一致時は 6 バイトへ切り詰め/ゼロ埋め
 */
// Deprecated: use '@/lib/utils/mac-address' instead
export { toMacUint8Array } from '@/lib/utils/mac-address';

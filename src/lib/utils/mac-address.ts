/**
 * MACアドレス変換ユーティリティ
 *
 * - number[] や string[]("0x12"/"12") を受け付けて Uint8Array(6) に正規化
 * - 値は 0-255 にクランプ、少数は切り捨て
 * - 長さ不一致時は 6 バイトへ切り詰め/ゼロ埋め
 */
export function toMacUint8Array(input: ReadonlyArray<number | string>): Uint8Array {
  const out = new Uint8Array(6);
  for (let i = 0; i < 6; i++) {
    const raw = input[i] ?? 0;
    let n: number;
    if (typeof raw === 'number') {
      n = raw;
    } else {
      const s = raw.trim().toLowerCase();
      // 0x 前置の16進 or 10進数文字列を許容。非数だった場合は16進として再解釈を試みる
      n = s.startsWith('0x') ? parseInt(s, 16) : Number.isNaN(Number(s)) ? parseInt(s, 16) : Number(s);
    }
    if (!Number.isFinite(n)) n = 0;
    n = Math.min(255, Math.max(0, Math.trunc(n)));
    out[i] = n;
  }
  return out;
}

//! SIMD版 MT19937 実装
//!
//! 4系統の乱数生成器を並列に持つSIMD最適化版Mt19937。
//! WASM SIMDの128bit幅ベクトル演算（v128）を活用し、
//! 4つの異なるSeedから同時に乱数列を生成することでスループットを向上させる。

#[cfg(target_arch = "wasm32")]
const N: usize = 624;
#[cfg(target_arch = "wasm32")]
const M: usize = 397;
#[cfg(target_arch = "wasm32")]
const MATRIX_A: u32 = 0x9908B0DF;
#[cfg(target_arch = "wasm32")]
const UPPER_MASK: u32 = 0x80000000;
#[cfg(target_arch = "wasm32")]
const LOWER_MASK: u32 = 0x7FFFFFFF;

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

/// SIMD版 MT19937 (4系統並列)
/// 4つの異なるSeedから同時に乱数列を生成
#[cfg(target_arch = "wasm32")]
pub struct Mt19937x4 {
    index: usize,
    /// state[i] は4系統分のstate[i]をv128で保持
    state: [v128; N],
}

#[cfg(target_arch = "wasm32")]
impl Mt19937x4 {
    /// 4つのSeedで初期化
    #[inline]
    pub fn new(seeds: [u32; 4]) -> Self {
        let mut state = [u32x4_splat(0); N];

        // state[0] = seeds
        state[0] = u32x4(seeds[0], seeds[1], seeds[2], seeds[3]);

        for i in 1..N {
            // state[i] = 1812433253 * (state[i-1] ^ (state[i-1] >> 30)) + i
            let prev = state[i - 1];
            let shifted = u32x4_shr(prev, 30);
            let xored = v128_xor(prev, shifted);

            // 乗算は要素ごとに実行（WASMにはu32x4_mulがないため展開）
            let vals = [
                u32x4_extract_lane::<0>(xored),
                u32x4_extract_lane::<1>(xored),
                u32x4_extract_lane::<2>(xored),
                u32x4_extract_lane::<3>(xored),
            ];

            state[i] = u32x4(
                1812433253u32.wrapping_mul(vals[0]).wrapping_add(i as u32),
                1812433253u32.wrapping_mul(vals[1]).wrapping_add(i as u32),
                1812433253u32.wrapping_mul(vals[2]).wrapping_add(i as u32),
                1812433253u32.wrapping_mul(vals[3]).wrapping_add(i as u32),
            );
        }

        Mt19937x4 { index: N, state }
    }

    /// 4系統同時に次の乱数を取得
    #[inline]
    pub fn next_u32x4(&mut self) -> v128 {
        if self.index >= N {
            self.twist();
        }

        let mut y = self.state[self.index];
        self.index += 1;

        // Tempering (SIMD版)
        y = v128_xor(y, u32x4_shr(y, 11));
        y = v128_xor(y, v128_and(u32x4_shl(y, 7), u32x4_splat(0x9D2C_5680)));
        y = v128_xor(y, v128_and(u32x4_shl(y, 15), u32x4_splat(0xEFC6_0000)));
        y = v128_xor(y, u32x4_shr(y, 18));

        y
    }

    #[inline]
    fn twist(&mut self) {
        let matrix_a = u32x4_splat(MATRIX_A);
        let upper_mask = u32x4_splat(UPPER_MASK);
        let lower_mask = u32x4_splat(LOWER_MASK);
        let one = u32x4_splat(1);

        for i in 0..N {
            let x = v128_or(
                v128_and(self.state[i], upper_mask),
                v128_and(self.state[(i + 1) % N], lower_mask),
            );

            let x_shr = u32x4_shr(x, 1);

            // x & 1 != 0 の場合のみ MATRIX_A を XOR
            let odd_mask = u32x4_eq(v128_and(x, one), one);
            let x_a = v128_xor(x_shr, v128_and(matrix_a, odd_mask));

            self.state[i] = v128_xor(self.state[(i + M) % N], x_a);
        }

        self.index = 0;
    }

    /// 4系統の結果を個別のu32として取得
    #[inline]
    pub fn extract_lanes(v: v128) -> [u32; 4] {
        [
            u32x4_extract_lane::<0>(v),
            u32x4_extract_lane::<1>(v),
            u32x4_extract_lane::<2>(v),
            u32x4_extract_lane::<3>(v),
        ]
    }
}

/// 非WASM環境用のフォールバック実装
#[cfg(not(target_arch = "wasm32"))]
pub struct Mt19937x4 {
    mts: [crate::mt19937::Mt19937; 4],
}

#[cfg(not(target_arch = "wasm32"))]
impl Mt19937x4 {
    pub fn new(seeds: [u32; 4]) -> Self {
        Mt19937x4 {
            mts: [
                crate::mt19937::Mt19937::new(seeds[0]),
                crate::mt19937::Mt19937::new(seeds[1]),
                crate::mt19937::Mt19937::new(seeds[2]),
                crate::mt19937::Mt19937::new(seeds[3]),
            ],
        }
    }

    /// 4系統同時に次の乱数を取得（配列として返す）
    pub fn next_u32x4(&mut self) -> [u32; 4] {
        [
            self.mts[0].next_u32(),
            self.mts[1].next_u32(),
            self.mts[2].next_u32(),
            self.mts[3].next_u32(),
        ]
    }

    /// フォールバック用：配列をそのまま返す
    pub fn extract_lanes(v: [u32; 4]) -> [u32; 4] {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937x4_matches_single_mt() {
        // 単体版Mt19937と結果が一致することを確認
        let seeds = [5489, 12345, 67890, 11111];

        let mut mt_simd = Mt19937x4::new(seeds);

        // 各系統の単体Mt19937を用意
        let mut mt_singles = [
            crate::mt19937::Mt19937::new(seeds[0]),
            crate::mt19937::Mt19937::new(seeds[1]),
            crate::mt19937::Mt19937::new(seeds[2]),
            crate::mt19937::Mt19937::new(seeds[3]),
        ];

        // 100回分の乱数を比較
        for _ in 0..100 {
            let simd_result = mt_simd.next_u32x4();
            let lanes = Mt19937x4::extract_lanes(simd_result);

            for j in 0..4 {
                let single_result = mt_singles[j].next_u32();
                assert_eq!(
                    lanes[j], single_result,
                    "Mismatch in lane {} for seed {}",
                    j, seeds[j]
                );
            }
        }
    }

    #[test]
    fn mt19937x4_iv_derivation() {
        // IV導出が正しく動作することを確認
        let seeds = [0, 1, 2, 3];
        let mut mt = Mt19937x4::new(seeds);

        // advances = 0 でIV取得
        let mut ivs = [[0u8; 6]; 4];
        for stat in 0..6 {
            let rand = mt.next_u32x4();
            let lanes = Mt19937x4::extract_lanes(rand);
            for i in 0..4 {
                ivs[i][stat] = (lanes[i] >> 27) as u8;
            }
        }

        // 各IVが0-31の範囲内であることを確認
        for i in 0..4 {
            for stat in 0..6 {
                assert!(ivs[i][stat] <= 31, "IV out of range: {}", ivs[i][stat]);
            }
        }
    }
}

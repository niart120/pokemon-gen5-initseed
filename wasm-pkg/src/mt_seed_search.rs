//! MT Seed 32bit全探索 実装
//!
//! MT消費数と検索条件（IVコード）が与えられたとき、
//! MT19937のSeed空間を探索し、所定のIVパターンを生成するMT Seedを検索する。

use crate::mt19937::Mt19937;
use crate::mt19937_simd::Mt19937x4;
use std::collections::HashSet;
use wasm_bindgen::prelude::*;

/// IVコード型（30bit圧縮表現）
/// 配置: [HP:5bit][Atk:5bit][Def:5bit][SpA:5bit][SpD:5bit][Spe:5bit]
pub type IvCode = u32;

/// IVセットをIVコードにエンコード
#[inline]
pub fn encode_iv_code(ivs: &[u8; 6]) -> IvCode {
    ((ivs[0] as u32) << 25)
        | ((ivs[1] as u32) << 20)
        | ((ivs[2] as u32) << 15)
        | ((ivs[3] as u32) << 10)
        | ((ivs[4] as u32) << 5)
        | (ivs[5] as u32)
}

/// IVコードをIVセットにデコード
#[inline]
pub fn decode_iv_code(code: IvCode) -> [u8; 6] {
    [
        ((code >> 25) & 0x1F) as u8,
        ((code >> 20) & 0x1F) as u8,
        ((code >> 15) & 0x1F) as u8,
        ((code >> 10) & 0x1F) as u8,
        ((code >> 5) & 0x1F) as u8,
        (code & 0x1F) as u8,
    ]
}

/// MT Seedから指定消費数後のIVセットを導出（単体版）
#[inline]
pub fn derive_iv_set(mt_seed: u32, advances: u32) -> [u8; 6] {
    let mut mt = Mt19937::new(mt_seed);

    // MT消費
    for _ in 0..advances {
        mt.next_u32();
    }

    // IV取得（上位5bit × 6ステータス）
    let mut ivs = [0u8; 6];
    for iv in ivs.iter_mut() {
        *iv = (mt.next_u32() >> 27) as u8;
    }

    ivs
}

/// 検索セグメント実行（単体版、HashSet使用）
///
/// # Arguments
/// * `start` - 検索開始Seed (inclusive)
/// * `end` - 検索終了Seed (inclusive)
/// * `advances` - MT消費数
/// * `target_codes` - 検索対象IVコードのHashSet
///
/// # Returns
/// マッチしたMT SeedとIVコードのペア配列
pub fn search_mt_seed_segment(
    start: u32,
    end: u32,
    advances: u32,
    target_codes: &HashSet<IvCode>,
) -> Vec<(u32, IvCode)> {
    let mut results = Vec::new();

    for seed in start..=end {
        let ivs = derive_iv_set(seed, advances);
        let code = encode_iv_code(&ivs);

        if target_codes.contains(&code) {
            results.push((seed, code));
        }
    }

    results
}

/// SIMD版検索セグメント実行
///
/// 4系統同時に探索を実行し、スループットを向上させる
///
/// # Arguments
/// * `start` - 検索開始Seed (inclusive)
/// * `end` - 検索終了Seed (inclusive)
/// * `advances` - MT消費数
/// * `target_codes` - 検索対象IVコードのHashSet
///
/// # Returns
/// マッチしたMT SeedとIVコードのペア配列
pub fn search_mt_seed_segment_simd(
    start: u32,
    end: u32,
    advances: u32,
    target_codes: &HashSet<IvCode>,
) -> Vec<(u32, IvCode)> {
    let mut results = Vec::new();

    // 4の倍数に切り下げ
    let aligned_start = start & !3;
    let mut seed = aligned_start;

    while seed <= end {
        // 4つのSeedを同時処理
        let seeds = [seed, seed + 1, seed + 2, seed + 3];
        let mut mt = Mt19937x4::new(seeds);

        // MT消費
        for _ in 0..advances {
            mt.next_u32x4();
        }

        // IV取得（6回の乱数取得）
        let mut iv_vals = [[0u8; 6]; 4];
        for stat in 0..6 {
            let rand = mt.next_u32x4();
            let lanes = Mt19937x4::extract_lanes(rand);
            for i in 0..4 {
                iv_vals[i][stat] = (lanes[i] >> 27) as u8;
            }
        }

        // 各系統の結果をチェック
        for (i, ivs) in iv_vals.iter().enumerate() {
            let current_seed = seed.wrapping_add(i as u32);

            // 範囲外チェック
            if current_seed < start || current_seed > end {
                continue;
            }

            let code = encode_iv_code(ivs);
            if target_codes.contains(&code) {
                results.push((current_seed, code));
            }
        }

        // 次の4つのSeedへ（オーバーフロー対策）
        seed = seed.wrapping_add(4);
        if seed < aligned_start {
            break; // オーバーフロー検出
        }
    }

    results
}

/// WASM公開関数: MT Seed検索セグメント実行
///
/// SIMD最適化版を使用して検索を実行
///
/// # Arguments
/// * `start` - 検索開始Seed (inclusive)
/// * `end` - 検索終了Seed (inclusive)
/// * `advances` - MT消費数
/// * `target_codes` - 検索対象IVコードのスライス
///
/// # Returns
/// フラット配列 [seed0, code0, seed1, code1, ...]
#[wasm_bindgen]
pub fn mt_seed_search_segment(
    start: u32,
    end: u32,
    advances: u32,
    target_codes: &[u32],
) -> Vec<u32> {
    // target_codesをHashSetに変換
    let target_set: HashSet<IvCode> = target_codes.iter().cloned().collect();

    // SIMD版で検索実行
    let results = search_mt_seed_segment_simd(start, end, advances, &target_set);

    // 結果をフラット配列で返す [seed0, code0, seed1, code1, ...]
    results
        .into_iter()
        .flat_map(|(seed, code)| [seed, code])
        .collect()
}

/// WASM公開関数: IVセット導出
///
/// MT SeedとMT消費数からIVセットを導出する
///
/// # Arguments
/// * `mt_seed` - MT Seed
/// * `advances` - MT消費数
///
/// # Returns
/// IVセット [HP, Atk, Def, SpA, SpD, Spe]
#[wasm_bindgen]
pub fn derive_iv_set_wasm(mt_seed: u32, advances: u32) -> Vec<u8> {
    derive_iv_set(mt_seed, advances).to_vec()
}

/// WASM公開関数: IVコードエンコード
///
/// IVセットをIVコードにエンコードする
///
/// # Arguments
/// * `ivs` - IVセット [HP, Atk, Def, SpA, SpD, Spe]
///
/// # Returns
/// IVコード (30bit)
#[wasm_bindgen]
pub fn encode_iv_code_wasm(ivs: &[u8]) -> u32 {
    if ivs.len() != 6 {
        return 0;
    }
    let arr: [u8; 6] = [ivs[0], ivs[1], ivs[2], ivs[3], ivs[4], ivs[5]];
    encode_iv_code(&arr)
}

/// WASM公開関数: IVコードデコード
///
/// IVコードをIVセットにデコードする
///
/// # Arguments
/// * `code` - IVコード (30bit)
///
/// # Returns
/// IVセット [HP, Atk, Def, SpA, SpD, Spe]
#[wasm_bindgen]
pub fn decode_iv_code_wasm(code: u32) -> Vec<u8> {
    decode_iv_code(code).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_iv_code() {
        let ivs = [31, 31, 31, 31, 31, 31]; // 6V
        let code = encode_iv_code(&ivs);
        let decoded = decode_iv_code(code);
        assert_eq!(ivs, decoded);

        let ivs2 = [0, 0, 0, 0, 0, 0]; // 0V
        let code2 = encode_iv_code(&ivs2);
        assert_eq!(code2, 0);
        let decoded2 = decode_iv_code(code2);
        assert_eq!(ivs2, decoded2);

        let ivs3 = [31, 0, 31, 31, 31, 31]; // 5V0A
        let code3 = encode_iv_code(&ivs3);
        let decoded3 = decode_iv_code(code3);
        assert_eq!(ivs3, decoded3);
    }

    #[test]
    fn test_derive_iv_set() {
        // 既知のSeedでIV導出をテスト
        let seed = 0;
        let ivs = derive_iv_set(seed, 0);

        // 各IVが0-31の範囲内であることを確認
        for iv in ivs.iter() {
            assert!(*iv <= 31, "IV out of range: {}", iv);
        }
    }

    #[test]
    fn test_search_mt_seed_segment() {
        // 既知のSeed/IVペアで検索テスト
        let seed = 12345u32;
        let ivs = derive_iv_set(seed, 0);
        let code = encode_iv_code(&ivs);

        let mut target_codes = HashSet::new();
        target_codes.insert(code);

        // 単体版
        let results = search_mt_seed_segment(seed - 10, seed + 10, 0, &target_codes);
        assert!(!results.is_empty(), "Should find the target seed");
        assert!(
            results.iter().any(|(s, _)| *s == seed),
            "Should find exact seed"
        );

        // SIMD版
        let results_simd = search_mt_seed_segment_simd(seed - 10, seed + 10, 0, &target_codes);
        assert!(!results_simd.is_empty(), "SIMD should find the target seed");
        assert!(
            results_simd.iter().any(|(s, _)| *s == seed),
            "SIMD should find exact seed"
        );
    }

    #[test]
    fn test_simd_matches_scalar() {
        // SIMD版と単体版の結果が一致することを確認
        let start = 0u32;
        let end = 1000u32;
        let advances = 0;

        // ターゲットIVコードを複数用意
        let mut target_codes = HashSet::new();
        for seed in (start..=end).step_by(100) {
            let ivs = derive_iv_set(seed, advances);
            let code = encode_iv_code(&ivs);
            target_codes.insert(code);
        }

        let results_scalar = search_mt_seed_segment(start, end, advances, &target_codes);
        let results_simd = search_mt_seed_segment_simd(start, end, advances, &target_codes);

        // 結果をソートして比較
        let mut scalar_sorted: Vec<_> = results_scalar.iter().map(|(s, _)| *s).collect();
        let mut simd_sorted: Vec<_> = results_simd.iter().map(|(s, _)| *s).collect();
        scalar_sorted.sort();
        simd_sorted.sort();

        assert_eq!(
            scalar_sorted, simd_sorted,
            "SIMD and scalar results should match"
        );
    }
}

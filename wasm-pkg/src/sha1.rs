use wasm_bindgen::prelude::*;

// =============================================================================
// SHA-1 ハッシュ値構造体
// =============================================================================

/// SHA-1ハッシュ値（5ワード）
///
/// SHA-1計算結果を保持し、LCG Seed や MT Seed への変換機能を提供する。
/// h2, h3, h4 は将来の拡張やデバッグ用に保持。
#[derive(Debug, Clone, Copy)]
pub struct HashValues {
    pub h0: u32,
    pub h1: u32,
    #[allow(dead_code)]
    pub h2: u32,
    #[allow(dead_code)]
    pub h3: u32,
    #[allow(dead_code)]
    pub h4: u32,
}

impl HashValues {
    #[inline]
    pub fn new(h0: u32, h1: u32, h2: u32, h3: u32, h4: u32) -> Self {
        Self { h0, h1, h2, h3, h4 }
    }

    /// 16進数文字列に変換（デバッグ用）
    #[allow(dead_code)]
    pub fn to_hex_string(&self) -> String {
        format!(
            "{:08x}{:08x}{:08x}{:08x}{:08x}",
            self.h0, self.h1, self.h2, self.h3, self.h4
        )
    }

    /// 64bit LCG Seedを計算
    #[inline]
    pub fn to_lcg_seed(&self) -> u64 {
        let h0_le = swap_bytes_32(self.h0) as u64;
        let h1_le = swap_bytes_32(self.h1) as u64;
        (h1_le << 32) | h0_le
    }

    /// 32bit MT Seedを計算（ポケモンBW/BW2用）
    #[inline]
    pub fn to_mt_seed(&self) -> u32 {
        calculate_pokemon_seed_from_hash(self.h0, self.h1)
    }
}

// =============================================================================
// SHA-1 計算
// =============================================================================

/// ポケモンBW/BW2特化SHA-1実装
/// 高速なSeed計算のためにカスタム最適化されたSHA-1関数
/// ポケモンBW/BW2のSHA-1実装
/// 16個の32bit値を受け取り、HashValuesを返す
///
/// このカスタム実装の特徴：
/// - ポケモン特有の16ワードメッセージに最適化
/// - TypeScript版と完全に同じ結果を保証
/// - WebAssemblyによる高速実行
#[inline]
pub fn calculate_pokemon_sha1(message: &[u32; 16]) -> HashValues {
    // SHA-1初期値
    const H0: u32 = 0x67452301;
    const H1: u32 = 0xEFCDAB89;
    const H2: u32 = 0x98BADCFE;
    const H3: u32 = 0x10325476;
    const H4: u32 = 0xC3D2E1F0;

    // 80ワードのメッセージスケジュール配列
    let mut w = [0u32; 80];

    // 最初の16ワードをコピー
    w[..16].copy_from_slice(message);

    // 残りの64ワードを計算
    for i in 16..80 {
        w[i] = left_rotate(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    }

    // メイン処理ループ
    let mut a = H0;
    let mut b = H1;
    let mut c = H2;
    let mut d = H3;
    let mut e = H4;

    for (i, &w_val) in w.iter().enumerate() {
        let (f, k) = match i {
            0..=19 => (choice(b, c, d), 0x5A827999),
            20..=39 => (parity(b, c, d), 0x6ED9EBA1),
            40..=59 => (majority(b, c, d), 0x8F1BBCDC),
            60..=79 => (parity(b, c, d), 0xCA62C1D6),
            _ => unreachable!(),
        };

        let temp = left_rotate(a, 5)
            .wrapping_add(f)
            .wrapping_add(e)
            .wrapping_add(k)
            .wrapping_add(w_val);

        e = d;
        d = c;
        c = left_rotate(b, 30);
        b = a;
        a = temp;
    }

    // 最終ハッシュ値計算
    HashValues::new(
        H0.wrapping_add(a),
        H1.wrapping_add(b),
        H2.wrapping_add(c),
        H3.wrapping_add(d),
        H4.wrapping_add(e),
    )
}

/// ポケモンBW/BW2用LCG計算
/// SHA-1ハッシュ値からTypeScript版と同じ方式で最終seedを計算
pub fn calculate_pokemon_seed_from_hash(h0: u32, h1: u32) -> u32 {
    // TypeScript版と同じバイトスワップとLCG計算
    let h0_le = swap_bytes_32(h0) as u64;
    let h1_le = swap_bytes_32(h1) as u64;

    // 64bit値を構築
    let lcg_seed = (h1_le << 32) | h0_le;

    // 64bit LCG演算
    let multiplier = 0x5D588B656C078965u64;
    let add_value = 0x269EC3u64;

    let seed = lcg_seed.wrapping_mul(multiplier).wrapping_add(add_value);

    // 上位32bitを取得
    ((seed >> 32) & 0xFFFFFFFF) as u32
}

/// WebAssembly向けバッチSHA-1計算エントリポイント
/// `messages` は 16 ワード単位（512bit）で並ぶフラットな配列である必要がある
#[wasm_bindgen]
pub fn sha1_hash_batch(messages: &[u32]) -> Vec<u32> {
    if messages.len() % 16 != 0 {
        wasm_bindgen::throw_str("sha1_hash_batch expects messages.len() % 16 == 0");
    }

    let message_count = messages.len() / 16;
    let mut output = Vec::with_capacity(message_count * 5);

    for chunk in messages.chunks_exact(16) {
        let mut block = [0u32; 16];
        block.copy_from_slice(chunk);
        let hash = calculate_pokemon_sha1(&block);
        output.extend_from_slice(&[hash.h0, hash.h1, hash.h2, hash.h3, hash.h4]);
    }

    output
}

/// SHA-1補助関数: Choice function
#[inline]
pub fn choice(x: u32, y: u32, z: u32) -> u32 {
    (x & y) | (!x & z)
}

/// SHA-1補助関数: Parity function
#[inline]
pub fn parity(x: u32, y: u32, z: u32) -> u32 {
    x ^ y ^ z
}

/// SHA-1補助関数: Majority function
#[inline]
pub fn majority(x: u32, y: u32, z: u32) -> u32 {
    (x & y) | (x & z) | (y & z)
}

/// 左回転関数
#[inline]
pub fn left_rotate(value: u32, amount: u32) -> u32 {
    (value << amount) | (value >> (32 - amount))
}

/// バイトスワップ関数（32bit）
/// TypeScript版と同じバイトスワップ処理を実行
pub fn swap_bytes_32(value: u32) -> u32 {
    ((value & 0xFF) << 24)
        | (((value >> 8) & 0xFF) << 16)
        | (((value >> 16) & 0xFF) << 8)
        | ((value >> 24) & 0xFF)
}

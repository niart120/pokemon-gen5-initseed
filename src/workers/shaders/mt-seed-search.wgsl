// MT Seed 32bit全探索用 WGSL シェーダー
//
// MT19937のSeed空間を全探索し、指定されたIVコードリストにマッチする
// MT Seedを検出する。各スレッドが1つのMT Seedを担当。

// === 定数 ===
const N: u32 = 624u;
const M: u32 = 397u;
const MATRIX_A: u32 = 0x9908B0DFu;
const UPPER_MASK: u32 = 0x80000000u;
const LOWER_MASK: u32 = 0x7FFFFFFFu;

// ワークグループサイズ（プレースホルダー、TypeScript側で置換）
const WORKGROUP_SIZE: u32 = WORKGROUP_SIZE_PLACEHOLDERu;

// === バインディング ===

// 検索パラメータ
struct SearchParams {
    start_seed: u32,      // 検索開始Seed
    end_seed: u32,        // 検索終了Seed（inclusive）
    advances: u32,        // MT消費数
    target_count: u32,    // 検索対象IVコード数
    max_results: u32,     // 最大結果数
    reserved0: u32,
    reserved1: u32,
    reserved2: u32,
}

// マッチ結果レコード
struct MatchRecord {
    seed: u32,
    iv_code: u32,
}

// 結果出力バッファ
struct ResultBuffer {
    match_count: atomic<u32>,
    records: array<MatchRecord>,
}

@group(0) @binding(0) var<uniform> params: SearchParams;
@group(0) @binding(1) var<storage, read> target_codes: array<u32>;
@group(0) @binding(2) var<storage, read_write> results: ResultBuffer;

// === MT19937 実装 ===
// privateメモリにstate配列を保持（約2.5KB）

var<private> mt_state: array<u32, 624>;
var<private> mt_index: u32;

// MT19937初期化
fn mt_init(seed: u32) {
    mt_state[0] = seed;
    for (var i = 1u; i < N; i++) {
        let prev = mt_state[i - 1u];
        mt_state[i] = 1812433253u * (prev ^ (prev >> 30u)) + i;
    }
    mt_index = N;
}

// MT19937 twist操作
fn mt_twist() {
    for (var i = 0u; i < N; i++) {
        let next_idx = (i + 1u) % N;
        let m_idx = (i + M) % N;
        
        let x = (mt_state[i] & UPPER_MASK) | (mt_state[next_idx] & LOWER_MASK);
        var x_a = x >> 1u;
        if ((x & 1u) != 0u) {
            x_a ^= MATRIX_A;
        }
        mt_state[i] = mt_state[m_idx] ^ x_a;
    }
    mt_index = 0u;
}

// MT19937 次の乱数を取得
fn mt_next() -> u32 {
    if (mt_index >= N) {
        mt_twist();
    }
    
    var y = mt_state[mt_index];
    mt_index++;
    
    // Tempering
    y ^= y >> 11u;
    y ^= (y << 7u) & 0x9D2C5680u;
    y ^= (y << 15u) & 0xEFC60000u;
    y ^= y >> 18u;
    
    return y;
}

// === IVコードエンコード ===
// 6ステータスのIV（各5bit）を30bitの整数にパック
// 配置: [HP:5bit][Atk:5bit][Def:5bit][SpA:5bit][SpD:5bit][Spe:5bit]

fn encode_iv_code(hp: u32, atk: u32, def: u32, spa: u32, spd: u32, spe: u32) -> u32 {
    return (hp << 25u) | (atk << 20u) | (def << 15u) | (spa << 10u) | (spd << 5u) | spe;
}

// === 線形探索 ===
// IVコード数は最大1024件のため、GPUの並列性を活かせば線形探索で十分高速
// 二分探索は分岐コストが高く、却って遅くなる可能性がある

fn linear_search(code: u32) -> bool {
    for (var i = 0u; i < params.target_count; i++) {
        if (target_codes[i] == code) {
            return true;
        }
    }
    return false;
}

// === メインエントリポイント ===

@compute @workgroup_size(WORKGROUP_SIZE_PLACEHOLDER)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // 担当するMT Seed を計算
    let seed = params.start_seed + global_id.x;
    
    // 範囲外チェック（オーバーフロー含む）
    if (seed < params.start_seed || seed > params.end_seed) {
        return;
    }
    
    // MT19937初期化
    mt_init(seed);
    
    // MT消費（advances回）
    for (var i = 0u; i < params.advances; i++) {
        let _ = mt_next();
    }
    
    // IV取得（6回の乱数取得、上位5bit）
    let hp  = mt_next() >> 27u;
    let atk = mt_next() >> 27u;
    let def = mt_next() >> 27u;
    let spa = mt_next() >> 27u;
    let spd = mt_next() >> 27u;
    let spe = mt_next() >> 27u;
    
    // IVコードにエンコード
    let code = encode_iv_code(hp, atk, def, spa, spd, spe);
    
    // 線形探索でマッチング判定
    if (linear_search(code)) {
        // アトミックにカウンタをインクリメントして結果を格納
        let idx = atomicAdd(&results.match_count, 1u);
        
        // バッファオーバーフロー防止
        if (idx < params.max_results) {
            results.records[idx].seed = seed;
            results.records[idx].iv_code = code;
        }
    }
}

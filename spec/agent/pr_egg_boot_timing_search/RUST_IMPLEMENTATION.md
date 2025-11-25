# Rust WASM 実装詳細設計

## 1. ファイル構成

```
wasm-pkg/src/
├── egg_boot_timing_search.rs  # 新規: メイン検索器
├── lib.rs                      # 更新: 新モジュールのエクスポート
├── egg_seed_enumerator.rs      # 既存: 流用
├── integrated_search.rs        # 既存: 参照
└── egg_iv.rs                   # 既存: 流用
```

## 2. 構造体定義

### 2.1 EggBootTimingSearcher

```rust
use crate::datetime_codes::{DateCodeGenerator, TimeCodeGenerator};
use crate::egg_iv::{
    GenerationConditions, GenerationConditionsJs, IndividualFilter, IndividualFilterJs,
};
use crate::egg_seed_enumerator::{EggSeedEnumerator, ParentsIVs, ParentsIVsJs};
use crate::integrated_search::generate_key_codes;
use crate::offset_calculator::GameMode;
use crate::sha1::{calculate_pokemon_sha1, swap_bytes_32};
use crate::sha1_simd::calculate_pokemon_sha1_simd;
use chrono::{Datelike, NaiveDate, Timelike};
use wasm_bindgen::prelude::*;

/// 孵化乱数起動時間検索器
#[wasm_bindgen]
pub struct EggBootTimingSearcher {
    // SHA-1計算用パラメータ（IntegratedSeedSearcherと共通）
    hardware: String,
    base_message: [u32; 16],
    key_codes: Vec<u32>,
    allowed_second_mask: Box<[bool; 86400]>,
    
    // 孵化条件パラメータ
    conditions: GenerationConditions,
    parents: ParentsIVs,
    filter: Option<IndividualFilter>,
    consider_npc_consumption: bool,
    game_mode: GameMode,
    
    // 消費範囲
    user_offset: u64,
    advance_count: u32,
}
```

### 2.2 EggBootTimingSearchResult

```rust
/// 検索結果1件（起動条件 + 個体情報）
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct EggBootTimingSearchResult {
    // 起動条件
    year: u32,
    month: u32,
    date: u32,
    hour: u32,
    minute: u32,
    second: u32,
    timer0: u32,
    vcount: u32,
    key_code: u32,
    
    // LCG Seed
    lcg_seed_high: u32,
    lcg_seed_low: u32,
    
    // 個体情報
    advance: u64,
    is_stable: bool,
    ivs: [u8; 6],
    nature: u8,
    gender: u8,
    ability: u8,
    shiny: u8,
    pid: u32,
    hp_type: u8,
    hp_power: u8,
    hp_known: bool,
}

#[wasm_bindgen]
impl EggBootTimingSearchResult {
    // ゲッター群
    #[wasm_bindgen(getter)]
    pub fn year(&self) -> u32 { self.year }
    
    #[wasm_bindgen(getter)]
    pub fn month(&self) -> u32 { self.month }
    
    #[wasm_bindgen(getter)]
    pub fn date(&self) -> u32 { self.date }
    
    #[wasm_bindgen(getter)]
    pub fn hour(&self) -> u32 { self.hour }
    
    #[wasm_bindgen(getter)]
    pub fn minute(&self) -> u32 { self.minute }
    
    #[wasm_bindgen(getter)]
    pub fn second(&self) -> u32 { self.second }
    
    #[wasm_bindgen(getter)]
    pub fn timer0(&self) -> u32 { self.timer0 }
    
    #[wasm_bindgen(getter)]
    pub fn vcount(&self) -> u32 { self.vcount }
    
    #[wasm_bindgen(getter = keyCode)]
    pub fn key_code(&self) -> u32 { self.key_code }
    
    #[wasm_bindgen(getter = lcgSeedHex)]
    pub fn lcg_seed_hex(&self) -> String {
        let seed = ((self.lcg_seed_high as u64) << 32) | (self.lcg_seed_low as u64);
        format!("{:016X}", seed)
    }
    
    #[wasm_bindgen(getter)]
    pub fn advance(&self) -> u64 { self.advance }
    
    #[wasm_bindgen(getter = isStable)]
    pub fn is_stable(&self) -> bool { self.is_stable }
    
    #[wasm_bindgen(getter)]
    pub fn ivs(&self) -> Vec<u8> { self.ivs.to_vec() }
    
    #[wasm_bindgen(getter)]
    pub fn nature(&self) -> u8 { self.nature }
    
    /// Gender: 0=Male, 1=Female, 2=Genderless
    #[wasm_bindgen(getter)]
    pub fn gender(&self) -> u8 { self.gender }
    
    /// Ability slot: 0=Ability1, 1=Ability2, 2=Hidden
    #[wasm_bindgen(getter)]
    pub fn ability(&self) -> u8 { self.ability }
    
    /// Shiny type: 0=Normal (not shiny), 1=Square shiny, 2=Star shiny
    #[wasm_bindgen(getter)]
    pub fn shiny(&self) -> u8 { self.shiny }
    
    #[wasm_bindgen(getter)]
    pub fn pid(&self) -> u32 { self.pid }
    
    #[wasm_bindgen(getter = hpType)]
    pub fn hp_type(&self) -> u8 { self.hp_type }
    
    #[wasm_bindgen(getter = hpPower)]
    pub fn hp_power(&self) -> u8 { self.hp_power }
    
    #[wasm_bindgen(getter = hpKnown)]
    pub fn hp_known(&self) -> bool { self.hp_known }
}
```

## 3. コンストラクタ実装

```rust
const EPOCH_2000_UNIX: i64 = 946684800;
const SECONDS_PER_DAY: i64 = 86_400;

#[derive(Clone, Copy)]
struct DailyTimeRangeConfig {
    hour_start: u32,
    hour_end: u32,
    minute_start: u32,
    minute_end: u32,
    second_start: u32,
    second_end: u32,
}

impl DailyTimeRangeConfig {
    fn new(
        hour_start: u32, hour_end: u32,
        minute_start: u32, minute_end: u32,
        second_start: u32, second_end: u32,
    ) -> Result<Self, JsValue> {
        // バリデーション（IntegratedSeedSearcherと同様）
        // ...
        Ok(DailyTimeRangeConfig { ... })
    }
}

fn build_allowed_second_mask(range: &DailyTimeRangeConfig) -> Box<[bool; 86400]> {
    // IntegratedSeedSearcherと同様
    // ...
}

#[wasm_bindgen]
impl EggBootTimingSearcher {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        // SHA-1 パラメータ
        mac: &[u8],
        nazo: &[u32],
        hardware: &str,
        key_input_mask: u32,
        frame: u32,
        
        // 時刻範囲
        hour_start: u32, hour_end: u32,
        minute_start: u32, minute_end: u32,
        second_start: u32, second_end: u32,
        
        // 孵化条件
        conditions: &GenerationConditionsJs,
        parents: &ParentsIVsJs,
        filter_js: Option<IndividualFilterJs>,
        consider_npc_consumption: bool,
        game_mode: GameMode,
        
        // 消費範囲
        user_offset: u64,
        advance_count: u32,
    ) -> Result<EggBootTimingSearcher, JsValue> {
        // バリデーション
        if mac.len() != 6 {
            return Err(JsValue::from_str("MAC address must be 6 bytes"));
        }
        if nazo.len() != 5 {
            return Err(JsValue::from_str("nazo must be 5 32-bit words"));
        }
        
        let time_range = DailyTimeRangeConfig::new(
            hour_start, hour_end,
            minute_start, minute_end,
            second_start, second_end,
        )?;
        
        let allowed_second_mask = build_allowed_second_mask(&time_range);
        
        match hardware {
            "DS" | "DS_LITE" | "3DS" => {}
            _ => return Err(JsValue::from_str("Hardware must be DS, DS_LITE, or 3DS")),
        }
        
        // 基本メッセージテンプレート構築（IntegratedSeedSearcherと同様）
        let mut base_message = [0u32; 16];
        
        for i in 0..5 {
            base_message[i] = swap_bytes_32(nazo[i]);
        }
        
        let mac_lower = ((mac[4] as u32) << 8) | (mac[5] as u32);
        base_message[6] = mac_lower;
        
        let mac_upper = (mac[0] as u32)
            | ((mac[1] as u32) << 8)
            | ((mac[2] as u32) << 16)
            | ((mac[3] as u32) << 24);
        let gx_stat = 0x06000000u32;
        let data7 = mac_upper ^ gx_stat ^ frame;
        base_message[7] = swap_bytes_32(data7);
        
        base_message[10] = 0x00000000;
        base_message[11] = 0x00000000;
        base_message[12] = 0;
        base_message[13] = 0x80000000;
        base_message[14] = 0x00000000;
        base_message[15] = 0x000001A0;
        
        let key_codes = generate_key_codes(key_input_mask);
        
        // 孵化条件の変換
        let internal_conditions = conditions.to_internal();
        let internal_parents = parents.to_internal();
        let internal_filter = filter_js.map(|f| f.to_internal());
        
        Ok(EggBootTimingSearcher {
            hardware: hardware.to_string(),
            base_message,
            key_codes,
            allowed_second_mask,
            conditions: internal_conditions,
            parents: internal_parents,
            filter: internal_filter,
            consider_npc_consumption,
            game_mode,
            user_offset,
            advance_count,
        })
    }
}
```

## 4. 検索メソッド実装

### 4.1 SIMD版検索

```rust
#[wasm_bindgen]
impl EggBootTimingSearcher {
    #[wasm_bindgen]
    #[inline(never)]
    pub fn search_eggs_integrated_simd(
        &self,
        year_start: u32,
        month_start: u32,
        date_start: u32,
        hour_start: u32,
        minute_start: u32,
        second_start: u32,
        range_seconds: u32,
        timer0_min: u32,
        timer0_max: u32,
        vcount_min: u32,
        vcount_max: u32,
    ) -> js_sys::Array {
        let results = js_sys::Array::new();
        
        // 開始日時をUnix時間に変換
        let start_datetime = match NaiveDate::from_ymd_opt(year_start as i32, month_start, date_start)
            .and_then(|date| date.and_hms_opt(hour_start, minute_start, second_start))
        {
            Some(datetime) => datetime,
            None => return results,
        };
        
        let start_unix = start_datetime.and_utc().timestamp();
        let base_seconds_since_2000 = start_unix - EPOCH_2000_UNIX;
        
        // 外側ループ: Timer0 × VCount × KeyCode
        for timer0 in timer0_min..=timer0_max {
            for vcount in vcount_min..=vcount_max {
                for &key_code in &self.key_codes {
                    // SIMD バッチ処理
                    let mut messages = [0u32; 64];
                    let mut batch_metadata: [(i64, u32, u32); 4] = [(0, 0, 0); 4];
                    let mut batch_len = 0usize;
                    
                    for second_offset in 0..range_seconds {
                        let current_seconds = base_seconds_since_2000 + second_offset as i64;
                        
                        // 日時コード計算（許可範囲チェック含む）
                        let (time_code, date_code) = match self.calculate_datetime_codes(current_seconds) {
                            Some(result) => result,
                            None => continue,
                        };
                        
                        // メッセージ構築
                        let message = self.build_message(timer0, vcount, date_code, time_code, key_code);
                        let base_idx = batch_len * 16;
                        messages[base_idx..base_idx + 16].copy_from_slice(&message);
                        batch_metadata[batch_len] = (current_seconds, timer0, vcount);
                        batch_len += 1;
                        
                        // 4件溜まったらSIMD処理
                        if batch_len == 4 {
                            self.process_simd_batch_egg(
                                &messages,
                                &batch_metadata,
                                batch_len,
                                key_code,
                                &results,
                            );
                            batch_len = 0;
                        }
                    }
                    
                    // 残りを処理
                    if batch_len > 0 {
                        self.process_simd_batch_egg(
                            &messages,
                            &batch_metadata,
                            batch_len,
                            key_code,
                            &results,
                        );
                    }
                }
            }
        }
        
        results
    }
    
    #[inline]
    fn process_simd_batch_egg(
        &self,
        messages: &[u32; 64],
        batch_metadata: &[(i64, u32, u32); 4],
        batch_size: usize,
        key_code: u32,
        results: &js_sys::Array,
    ) {
        if batch_size == 0 {
            return;
        }
        
        // SIMD または スカラー SHA-1 計算
        let hash_results = if batch_size == 4 {
            calculate_pokemon_sha1_simd(messages)
        } else {
            // スカラーフォールバック
            let mut scalar_results = [0u32; 20];
            for i in 0..batch_size {
                let mut single_message = [0u32; 16];
                let base_idx = i * 16;
                single_message.copy_from_slice(&messages[base_idx..base_idx + 16]);
                let (h0, h1, h2, h3, h4) = calculate_pokemon_sha1(&single_message);
                scalar_results[i * 5] = h0;
                scalar_results[i * 5 + 1] = h1;
                scalar_results[i * 5 + 2] = h2;
                scalar_results[i * 5 + 3] = h3;
                scalar_results[i * 5 + 4] = h4;
            }
            scalar_results
        };
        
        // 各LCG Seedに対して個体検索
        for i in 0..batch_size {
            let h0 = hash_results[i * 5];
            let h1 = hash_results[i * 5 + 1];
            let lcg_seed = crate::sha1::calculate_lcg_seed_from_hash(h0, h1);
            
            let (current_seconds, timer0, vcount) = batch_metadata[i];
            
            // 日時情報を取得
            let datetime = match self.generate_display_datetime(current_seconds) {
                Some(dt) => dt,
                None => continue,
            };
            
            // EggSeedEnumeratorで個体検索
            self.enumerate_eggs_for_seed(
                lcg_seed,
                datetime,
                timer0,
                vcount,
                key_code,
                results,
            );
        }
    }
    
    fn enumerate_eggs_for_seed(
        &self,
        lcg_seed: u64,
        datetime: (u32, u32, u32, u32, u32, u32),
        timer0: u32,
        vcount: u32,
        key_code: u32,
        results: &js_sys::Array,
    ) {
        let (year, month, date, hour, minute, second) = datetime;
        
        // EggSeedEnumeratorを作成
        let mut enumerator = EggSeedEnumerator::new(
            lcg_seed,
            self.user_offset,
            self.advance_count,
            self.conditions.clone(),
            self.parents.clone(),
            self.filter.clone(),
            self.consider_npc_consumption,
            self.game_mode,
        );
        
        // 条件に合う個体を列挙
        while let Ok(Some(egg_data)) = enumerator.next_egg() {
            let result = self.create_result(
                year, month, date, hour, minute, second,
                timer0, vcount, key_code,
                lcg_seed,
                &egg_data,
            );
            results.push(&JsValue::from(result));
        }
    }
    
    fn create_result(
        &self,
        year: u32, month: u32, date: u32,
        hour: u32, minute: u32, second: u32,
        timer0: u32, vcount: u32, key_code: u32,
        lcg_seed: u64,
        egg_data: &crate::egg_seed_enumerator::EnumeratedEggData,
    ) -> EggBootTimingSearchResult {
        let (hp_type, hp_power, hp_known) = match egg_data.egg.hidden_power {
            crate::egg_iv::HiddenPowerInfo::Known { r#type, power } => {
                (r#type as u8, power, true)
            }
            crate::egg_iv::HiddenPowerInfo::Unknown => (0, 0, false),
        };
        
        EggBootTimingSearchResult {
            year,
            month,
            date,
            hour,
            minute,
            second,
            timer0,
            vcount,
            key_code,
            lcg_seed_high: (lcg_seed >> 32) as u32,
            lcg_seed_low: lcg_seed as u32,
            advance: egg_data.advance,
            is_stable: egg_data.is_stable,
            ivs: egg_data.egg.ivs,
            nature: egg_data.egg.nature as u8,
            gender: match egg_data.egg.gender {
                crate::egg_iv::Gender::Male => 0,
                crate::egg_iv::Gender::Female => 1,
                crate::egg_iv::Gender::Genderless => 2,
            },
            ability: egg_data.egg.ability as u8,
            shiny: egg_data.egg.shiny as u8,
            pid: egg_data.egg.pid,
            hp_type,
            hp_power,
            hp_known,
        }
    }
    
    // ヘルパーメソッド（IntegratedSeedSearcherと共通化可能）
    #[inline(always)]
    fn calculate_datetime_codes(&self, seconds_since_2000: i64) -> Option<(u32, u32)> {
        if seconds_since_2000 < 0 {
            return None;
        }
        
        let seconds_of_day = (seconds_since_2000 % SECONDS_PER_DAY) as u32;
        if !self.is_second_allowed(seconds_of_day) {
            return None;
        }
        let date_index = (seconds_since_2000 / SECONDS_PER_DAY) as u32;
        
        let time_code = TimeCodeGenerator::get_time_code_for_hardware(seconds_of_day, &self.hardware);
        let date_code = DateCodeGenerator::get_date_code(date_index);
        
        Some((time_code, date_code))
    }
    
    #[inline(always)]
    fn is_second_allowed(&self, second_of_day: u32) -> bool {
        self.allowed_second_mask[second_of_day as usize]
    }
    
    fn generate_display_datetime(&self, seconds_since_2000: i64) -> Option<(u32, u32, u32, u32, u32, u32)> {
        let result_datetime = chrono::DateTime::from_timestamp(seconds_since_2000 + EPOCH_2000_UNIX, 0)?
            .naive_utc();
        
        Some((
            result_datetime.year() as u32,
            result_datetime.month(),
            result_datetime.day(),
            result_datetime.hour(),
            result_datetime.minute(),
            result_datetime.second(),
        ))
    }
    
    #[inline(always)]
    fn build_message(
        &self,
        timer0: u32,
        vcount: u32,
        date_code: u32,
        time_code: u32,
        key_code: u32,
    ) -> [u32; 16] {
        let mut message = self.base_message;
        message[5] = swap_bytes_32((vcount << 16) | timer0);
        message[8] = date_code;
        message[9] = time_code;
        message[12] = swap_bytes_32(key_code);
        message
    }
}
```

## 5. lib.rs 更新

```rust
// wasm-pkg/src/lib.rs に追加

mod egg_boot_timing_search;

// Re-export
pub use egg_boot_timing_search::{EggBootTimingSearcher, EggBootTimingSearchResult};
```

## 6. テストケース

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_searcher() -> EggBootTimingSearcher {
        let mac = [0x00, 0x1B, 0x2C, 0x3D, 0x4E, 0x5F];
        let nazo = [0x02215F10, 0x02761150, 0x00000000, 0x00000000, 0x02761150];
        
        let conditions = GenerationConditionsJs::new();
        let parents = ParentsIVsJs::new();
        
        EggBootTimingSearcher::new(
            &mac,
            &nazo,
            "DS",
            0,     // no key input
            8,     // frame
            0, 23, // hour
            0, 59, // minute
            0, 59, // second
            &conditions,
            &parents,
            None,  // no filter
            false, // no NPC consumption
            GameMode::BwContinue,
            0,     // user_offset
            100,   // advance_count
        ).unwrap()
    }
    
    #[test]
    fn test_constructor_validation() {
        // MAC アドレス長エラー
        let result = EggBootTimingSearcher::new(
            &[0x00, 0x1B, 0x2C], // 3 bytes
            &[0; 5],
            "DS",
            0, 8,
            0, 23, 0, 59, 0, 59,
            &GenerationConditionsJs::new(),
            &ParentsIVsJs::new(),
            None, false, GameMode::BwContinue,
            0, 100,
        );
        assert!(result.is_err());
    }
    
    #[test]
    fn test_search_returns_results() {
        let searcher = create_test_searcher();
        
        let results = searcher.search_eggs_integrated_simd(
            2025, 1, 1,
            12, 0, 0,
            60,        // 1 minute range
            0x0C79, 0x0C79,
            0x60, 0x60,
        );
        
        // 結果が配列として返される
        assert!(results.length() >= 0);
    }
    
    #[test]
    fn test_simd_scalar_consistency() {
        let searcher = create_test_searcher();
        
        // 4件以上のケースでSIMD/スカラーの結果が一致することを確認
        // （実装完了後に詳細テスト追加）
    }
}
```

## 7. 注意事項

### 7.1 パフォーマンス

- `EggSeedEnumerator` の生成コストを最小化するため、内部で再利用可能な部分を検討
- 大量結果時はストリーミング返却を検討

### 7.2 メモリ

- 結果配列は逐次的に `js_sys::Array::push` で追加
- 検索終了時にRust側のメモリは自動解放

### 7.3 エラーハンドリング

- `EggSeedEnumerator::next_egg()` のエラーは現状スキップ
- 将来的にはエラー情報の集約を検討

## 8. 共通化の余地

`IntegratedSeedSearcher` と `EggBootTimingSearcher` で以下の処理が重複:

1. `DailyTimeRangeConfig` 構造体
2. `build_allowed_second_mask` 関数
3. `build_message` メソッド
4. `calculate_datetime_codes` メソッド
5. `generate_display_datetime` メソッド

→ 将来的に `search_utils.rs` などに共通モジュールとして切り出すことを検討

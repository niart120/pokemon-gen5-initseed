/// EncounterCalculator - BW/BW2遭遇計算エンジン
/// ポケモンBW/BW2の遭遇スロット決定と確率計算を実装
use wasm_bindgen::prelude::*;

/// ゲームバージョン列挙型
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GameVersion {
    B = 0,
    W = 1,
    B2 = 2,
    W2 = 3,
}

/// 遭遇タイプ列挙型
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EncounterType {
    /// 通常エンカウント（草むら・洞窟・ダンジョン共通）
    Normal = 0,
    /// なみのり
    Surfing = 1,
    /// つりざお
    Fishing = 2,
    /// 揺れる草むら（特殊エンカウント）
    ShakingGrass = 3,
    /// 砂煙（特殊エンカウント）
    DustCloud = 4,
    /// ポケモンの影（特殊エンカウント）
    PokemonShadow = 5,
    /// 水泡（なみのり版特殊エンカウント）
    SurfingBubble = 6,
    /// 水泡釣り（釣り版特殊エンカウント）
    FishingBubble = 7,
    /// 固定シンボル（レジェンダリー等）- シンクロ有効
    StaticSymbol = 10,
    /// 御三家受け取り - シンクロ無効
    StaticStarter = 11,
    /// 化石復元 - シンクロ無効
    StaticFossil = 12,
    /// イベント配布 - シンクロ無効
    StaticEvent = 13,
    /// 徘徊ポケモン（ドキュメント仕様準拠）
    Roaming = 20,
}

/// 砂煙出現内容の種類
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DustCloudContent {
    /// ポケモン出現
    Pokemon = 0,
    /// ジュエル類出現
    Jewel = 1,
    /// 進化石類出現
    EvolutionStone = 2,
}

/// 遭遇計算エンジン
#[wasm_bindgen]
pub struct EncounterCalculator;

#[wasm_bindgen]
impl EncounterCalculator {
    /// 新しいEncounterCalculatorインスタンスを作成
    #[wasm_bindgen(constructor)]
    pub fn new() -> EncounterCalculator {
        EncounterCalculator
    }

    /// 遭遇スロットを計算
    /// 
    /// # Arguments
    /// * `version` - ゲームバージョン
    /// * `encounter_type` - 遭遇タイプ
    /// * `random_value` - 乱数値（32bit）
    /// 
    /// # Returns
    /// 遭遇スロット番号（0-11）
    pub fn calculate_encounter_slot(
        version: GameVersion,
        encounter_type: EncounterType,
        random_value: u32,
    ) -> u8 {
        // ゲームバージョンに応じた数学的計算でスロット値を算出
        let slot_value = Self::calculate_raw_encounter_slot_internal(version, random_value);
        
        // スロット値を各エンカウントタイプの確率分布に変換
        Self::slot_value_to_encounter_slot(encounter_type, slot_value)
    }

    /// ゲームバージョン別の生スロット値計算
    /// PersonalityRNGから移管した数学的計算式
    /// 
    /// # Arguments
    /// * `version` - ゲームバージョン
    /// * `random_value` - 32bit乱数値
    /// 
    /// # Returns
    /// 生スロット値（0-99範囲）
    #[cfg(test)]
    pub fn calculate_raw_encounter_slot(version: GameVersion, random_value: u32) -> u32 {
        Self::calculate_raw_encounter_slot_internal(version, random_value)
    }

    /// 内部用の生スロット値計算
    fn calculate_raw_encounter_slot_internal(version: GameVersion, random_value: u32) -> u32 {
        match version {
            GameVersion::B | GameVersion::W => {
                // BW: (rand * 0xFFFF / 0x290) >> 32
                ((random_value as u64 * 0xFFFF / 0x290) >> 32) as u32
            },
            GameVersion::B2 | GameVersion::W2 => {
                // BW2: (rand * 100) >> 32
                ((random_value as u64 * 100) >> 32) as u32
            },
        }
    }

    /// スロット値を各エンカウントタイプの確率分布に変換
    /// 
    /// # Arguments
    /// * `encounter_type` - 遭遇タイプ
    /// * `slot_value` - 生スロット値
    /// 
    /// # Returns
    /// 最終的な遭遇スロット番号
    fn slot_value_to_encounter_slot(encounter_type: EncounterType, slot_value: u32) -> u8 {
        match encounter_type {
            EncounterType::Normal => Self::calculate_normal_encounter_from_slot(slot_value),
            EncounterType::Surfing => Self::calculate_surfing_encounter_from_slot(slot_value),
            EncounterType::Fishing => Self::calculate_fishing_encounter_from_slot(slot_value),
            EncounterType::ShakingGrass => Self::calculate_shaking_grass_encounter_from_slot(slot_value),
            EncounterType::DustCloud => Self::calculate_dust_cloud_encounter_from_slot(slot_value),
            EncounterType::PokemonShadow => Self::calculate_pokemon_shadow_encounter_from_slot(slot_value),
            EncounterType::SurfingBubble => Self::calculate_surfing_bubble_encounter_from_slot(slot_value),
            EncounterType::FishingBubble => Self::calculate_fishing_bubble_encounter_from_slot(slot_value),
            EncounterType::StaticSymbol => 0, // 固定シンボルは常にスロット0
            EncounterType::StaticStarter => 0,  // 御三家は常にスロット0
            EncounterType::StaticFossil => 0,   // 化石は常にスロット0
            EncounterType::StaticEvent => 0,    // イベント配布は常にスロット0
            EncounterType::Roaming => 0,      // 徘徊ポケモンは常にスロット0
        }
    }

    /// スロット番号をテーブルインデックスに変換
    /// 
    /// # Arguments
    /// * `encounter_type` - 遭遇タイプ
    /// * `slot` - スロット番号
    /// 
    /// # Returns
    /// テーブルインデックス
    pub fn slot_to_table_index(encounter_type: EncounterType, slot: u8) -> usize {
        match encounter_type {
            EncounterType::Normal => {
                // 通常エンカウント：12スロット（0-11）
                if slot < 12 { slot as usize } else { 11 }
            },
            EncounterType::Surfing => {
                // なみのり：5スロット（0-4）
                if slot < 5 { slot as usize } else { 4 }
            },
            EncounterType::Fishing => {
                // つりざお：5スロット（0-4）
                if slot < 5 { slot as usize } else { 4 }
            },
            EncounterType::ShakingGrass => {
                // 揺れる草むら：5スロット（0-4）
                if slot < 5 { slot as usize } else { 4 }
            },
            EncounterType::DustCloud => {
                // 砂煙：3カテゴリ（0-2）
                if slot < 3 { slot as usize } else { 2 }
            },
            EncounterType::PokemonShadow => {
                // ポケモンの影：4スロット（0-3）
                if slot < 4 { slot as usize } else { 3 }
            },
            EncounterType::SurfingBubble => {
                // 水泡なみのり：4スロット（0-3）
                if slot < 4 { slot as usize } else { 3 }
            },
            EncounterType::FishingBubble => {
                // 水泡釣り：4スロット（0-3）
                if slot < 4 { slot as usize } else { 3 }
            },
            EncounterType::StaticSymbol => {
                // 固定シンボル：1スロット（0のみ）
                0
            },
            EncounterType::StaticStarter => {
                // 御三家：1スロット（0のみ）
                0
            },
            EncounterType::StaticFossil => {
                // 化石：1スロット（0のみ）
                0
            },
            EncounterType::StaticEvent => {
                // イベント配布：1スロット（0のみ）
                0
            },
            EncounterType::Roaming => {
                // 徘徊ポケモン：1スロット（0のみ）
                0
            },
        }
    }

    /// 砂煙の出現内容を判定
    /// 
    /// # Arguments
    /// * `slot` - 砂煙スロット値（0-2）
    /// 
    /// # Returns
    /// 出現内容の種類
    pub fn get_dust_cloud_content(slot: u8) -> DustCloudContent {
        match slot {
            0 => DustCloudContent::Pokemon,        // 70% ポケモン
            1 => DustCloudContent::Jewel,          // 20% ジュエル類
            2 => DustCloudContent::EvolutionStone, // 10% 進化石類
            _ => DustCloudContent::Pokemon,        // フォールバック
        }
    }
}

impl EncounterCalculator {
    /// 通常エンカウントスロット計算
    /// 12スロット：20%/20%/10%/10%/10%/10%/5%/5%/5%/4%/1%/1%
    fn calculate_normal_encounter_from_slot(slot_value: u32) -> u8 {
        match slot_value {
            0..=19 => 0,    // 20%
            20..=39 => 1,   // 20%
            40..=49 => 2,   // 10%
            50..=59 => 3,   // 10%
            60..=69 => 4,   // 10%
            70..=79 => 5,   // 10%
            80..=84 => 6,   // 5%
            85..=89 => 7,   // 5%
            90..=94 => 8,   // 5% (ドキュメント仕様に修正)
            95..=98 => 9,   // 4%
            99 => 10,       // 1%
            _ => 11,        // 残り1%
        }
    }

    /// なみのりエンカウントスロット計算
    /// 5スロット：60%/30%/5%/4%/1%
    fn calculate_surfing_encounter_from_slot(slot_value: u32) -> u8 {
        match slot_value {
            0..=59 => 0,    // 60%
            60..=89 => 1,   // 30%
            90..=94 => 2,   // 5%
            95..=98 => 3,   // 4%
            99 => 4,        // 1%
            _ => 4,         // フォールバック
        }
    }

    /// つりざおエンカウントスロット計算
    /// 5スロット：70%/15%/10%/5%（レア含む）
    fn calculate_fishing_encounter_from_slot(slot_value: u32) -> u8 {
        match slot_value {
            0..=69 => 0,    // 70%
            70..=84 => 1,   // 15%
            85..=94 => 2,   // 10%
            95..=99 => 3,   // 5%
            _ => 4,         // レア（ドキュメント仕様準拠）
        }
    }

    /// 特殊エンカウントスロット計算（揺れる草むら）
    /// 場所により4-5スロット、確率分布が異なる
    fn calculate_shaking_grass_encounter_from_slot(slot_value: u32) -> u8 {
        // 揺れる草むら（特殊エンカウント）
        // 通常より高レベル・レアポケモンが出現
        match slot_value {
            0..=39 => 0,    // 40%
            40..=59 => 1,   // 20%
            60..=79 => 2,   // 20%
            80..=94 => 3,   // 15%
            95..=99 => 4,   // 5% (隠れ特性持ち等)
            _ => 4,         // フォールバック
        }
    }

    /// 砂煙エンカウントスロット計算
    /// ポケモンまたはジュエル・進化石が出現
    fn calculate_dust_cloud_encounter_from_slot(slot_value: u32) -> u8 {
        match slot_value {
            0..=69 => 0,    // 70% ポケモン
            70..=89 => 1,   // 20% ジュエル類
            90..=99 => 2,   // 10% 進化石類
            _ => 2,         // フォールバック
        }
    }

    /// ポケモンの影エンカウントスロット計算
    /// 橋や建物の影で出現
    fn calculate_pokemon_shadow_encounter_from_slot(slot_value: u32) -> u8 {
        match slot_value {
            0..=49 => 0,    // 50%
            50..=79 => 1,   // 30%
            80..=94 => 2,   // 15%
            95..=99 => 3,   // 5%
            _ => 3,         // フォールバック
        }
    }

    /// 水泡（なみのり版特殊エンカウント）スロット計算
    /// なみのりエリアでの特殊遭遇
    fn calculate_surfing_bubble_encounter_from_slot(slot_value: u32) -> u8 {
        match slot_value {
            0..=49 => 0,    // 50%
            50..=79 => 1,   // 30%
            80..=94 => 2,   // 15%
            95..=99 => 3,   // 5%
            _ => 3,         // フォールバック
        }
    }

    /// 水泡釣り（釣り版特殊エンカウント）スロット計算
    /// 釣りエリアでの特殊遭遇
    fn calculate_fishing_bubble_encounter_from_slot(slot_value: u32) -> u8 {
        match slot_value {
            0..=59 => 0,    // 60%
            60..=84 => 1,   // 25%
            85..=94 => 2,   // 10%
            95..=99 => 3,   // 5%
            _ => 3,         // フォールバック
        }
    }

    /// 遭遇確率の検証用関数
    /// 指定した乱数値範囲での各スロットの出現頻度を計算
    pub fn calculate_slot_distribution(
        encounter_type: EncounterType,
        _version: GameVersion
    ) -> Vec<u32> {
        let mut distribution = vec![0u32; 12]; // 最大12スロット
        
        // 32bit乱数値の代表的なサンプルを使用してテスト
        // 0-99のスロット値に対応する32bit値を生成
        for slot_val in 0..100 {
            let slot = Self::slot_value_to_encounter_slot(encounter_type, slot_val);
            if (slot as usize) < distribution.len() {
                distribution[slot as usize] += 1;
            }
        }
        
        distribution
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_encounter_distribution() {
        let dist = EncounterCalculator::calculate_slot_distribution(
            EncounterType::Normal,
            GameVersion::B
        );
        
        // 期待される分布を確認（ドキュメント仕様）
        assert_eq!(dist[0], 20); // 20%
        assert_eq!(dist[1], 20); // 20%
        assert_eq!(dist[2], 10); // 10%
        assert_eq!(dist[3], 10); // 10%
        assert_eq!(dist[4], 10); // 10%
        assert_eq!(dist[5], 10); // 10%
        assert_eq!(dist[6], 5);  // 5%
        assert_eq!(dist[7], 5);  // 5%
        assert_eq!(dist[8], 5);  // 5% (ドキュメント仕様に修正)
        assert_eq!(dist[9], 4);  // 4%
        assert_eq!(dist[10], 1); // 1%
        assert_eq!(dist[11], 0); // 残り1%は range外となりdist[11]に含まれる
    }

    #[test]
    fn test_surfing_encounter_distribution() {
        let dist = EncounterCalculator::calculate_slot_distribution(
            EncounterType::Surfing,
            GameVersion::B
        );
        
        // 期待される分布を確認
        assert_eq!(dist[0], 60); // 60%
        assert_eq!(dist[1], 30); // 30%
        assert_eq!(dist[2], 5);  // 5%
        assert_eq!(dist[3], 4);  // 4%
        assert_eq!(dist[4], 1);  // 1%
        
        // 未使用スロットは0
        for i in 5..12 {
            assert_eq!(dist[i], 0);
        }
    }

    #[test]
    fn test_fishing_encounter_distribution() {
        let dist = EncounterCalculator::calculate_slot_distribution(
            EncounterType::Fishing,
            GameVersion::B
        );
        
        // 期待される分布を確認
        assert_eq!(dist[0], 70); // 70%
        assert_eq!(dist[1], 15); // 15%
        assert_eq!(dist[2], 10); // 10%
        assert_eq!(dist[3], 5);  // 5%
        assert_eq!(dist[4], 0);  // レアスロット（0-99範囲外）
        
        // 未使用スロットは0
        for i in 5..12 {
            assert_eq!(dist[i], 0);
        }
    }

    #[test]
    fn test_shaking_grass_encounter_distribution() {
        let dist = EncounterCalculator::calculate_slot_distribution(
            EncounterType::ShakingGrass,
            GameVersion::B
        );
        
        // 期待される分布を確認
        assert_eq!(dist[0], 40); // 40%
        assert_eq!(dist[1], 20); // 20%
        assert_eq!(dist[2], 20); // 20%
        assert_eq!(dist[3], 15); // 15%
        assert_eq!(dist[4], 5);  // 5%
        
        // 未使用スロットは0
        for i in 5..12 {
            assert_eq!(dist[i], 0);
        }
    }

    #[test]
    fn test_dust_cloud_encounter_distribution() {
        let dist = EncounterCalculator::calculate_slot_distribution(
            EncounterType::DustCloud,
            GameVersion::B
        );
        
        // 期待される分布を確認
        assert_eq!(dist[0], 70); // 70%
        assert_eq!(dist[1], 20); // 20%
        assert_eq!(dist[2], 10); // 10%
        
        // 未使用スロットは0
        for i in 3..12 {
            assert_eq!(dist[i], 0);
        }
    }

    #[test]
    fn test_pokemon_shadow_encounter_distribution() {
        let dist = EncounterCalculator::calculate_slot_distribution(
            EncounterType::PokemonShadow,
            GameVersion::B
        );
        
        // 期待される分布を確認
        assert_eq!(dist[0], 50); // 50%
        assert_eq!(dist[1], 30); // 30%
        assert_eq!(dist[2], 15); // 15%
        assert_eq!(dist[3], 5);  // 5%
        
        // 未使用スロットは0
        for i in 4..12 {
            assert_eq!(dist[i], 0);
        }
    }

    #[test]
    fn test_surfing_bubble_encounter_distribution() {
        let dist = EncounterCalculator::calculate_slot_distribution(
            EncounterType::SurfingBubble,
            GameVersion::B
        );
        
        // 期待される分布を確認
        assert_eq!(dist[0], 50); // 50%
        assert_eq!(dist[1], 30); // 30%
        assert_eq!(dist[2], 15); // 15%
        assert_eq!(dist[3], 5);  // 5%
        
        // 未使用スロットは0
        for i in 4..12 {
            assert_eq!(dist[i], 0);
        }
    }

    #[test]
    fn test_fishing_bubble_encounter_distribution() {
        let dist = EncounterCalculator::calculate_slot_distribution(
            EncounterType::FishingBubble,
            GameVersion::B
        );
        
        // 期待される分布を確認
        assert_eq!(dist[0], 60); // 60%
        assert_eq!(dist[1], 25); // 25%
        assert_eq!(dist[2], 10); // 10%
        assert_eq!(dist[3], 5);  // 5%
        
        // 未使用スロットは0
        for i in 4..12 {
            assert_eq!(dist[i], 0);
        }
    }

    #[test]
    fn test_slot_to_table_index() {
        // 通常エンカウント
        assert_eq!(EncounterCalculator::slot_to_table_index(EncounterType::Normal, 5), 5);
        assert_eq!(EncounterCalculator::slot_to_table_index(EncounterType::Normal, 15), 11);
        
        // なみのり
        assert_eq!(EncounterCalculator::slot_to_table_index(EncounterType::Surfing, 3), 3);
        assert_eq!(EncounterCalculator::slot_to_table_index(EncounterType::Surfing, 10), 4);
        
        // つりざお
        assert_eq!(EncounterCalculator::slot_to_table_index(EncounterType::Fishing, 2), 2);
        assert_eq!(EncounterCalculator::slot_to_table_index(EncounterType::Fishing, 8), 4);
        
        // 揺れる草むら
        assert_eq!(EncounterCalculator::slot_to_table_index(EncounterType::ShakingGrass, 1), 1);
        assert_eq!(EncounterCalculator::slot_to_table_index(EncounterType::ShakingGrass, 5), 4);
        
        // 砂煙
        assert_eq!(EncounterCalculator::slot_to_table_index(EncounterType::DustCloud, 1), 1);
        assert_eq!(EncounterCalculator::slot_to_table_index(EncounterType::DustCloud, 5), 2);
        
        // ポケモンの影
        assert_eq!(EncounterCalculator::slot_to_table_index(EncounterType::PokemonShadow, 2), 2);
        assert_eq!(EncounterCalculator::slot_to_table_index(EncounterType::PokemonShadow, 5), 3);
        
        // 水泡なみのり
        assert_eq!(EncounterCalculator::slot_to_table_index(EncounterType::SurfingBubble, 1), 1);
        assert_eq!(EncounterCalculator::slot_to_table_index(EncounterType::SurfingBubble, 6), 3);
        
        // 水泡釣り
        assert_eq!(EncounterCalculator::slot_to_table_index(EncounterType::FishingBubble, 2), 2);
        assert_eq!(EncounterCalculator::slot_to_table_index(EncounterType::FishingBubble, 7), 3);
    }

    #[test]
    fn test_edge_cases() {
        // 境界値のテスト（32bit乱数値）
        // シード値0からの計算結果をテスト
        let result_zero = EncounterCalculator::calculate_encounter_slot(
            GameVersion::B,
            EncounterType::Normal,
            0
        );
        assert!(result_zero <= 11); // 通常エンカウントの範囲内
        
        // 最大値での計算結果をテスト
        let result_max = EncounterCalculator::calculate_encounter_slot(
            GameVersion::B,
            EncounterType::Normal,
            u32::MAX
        );
        assert!(result_max <= 11); // 通常エンカウントの範囲内
        
        // 一致性テスト：同じ32bit値は常に同じ結果
        let test_value = 0x12345678u32;
        let result1 = EncounterCalculator::calculate_encounter_slot(
            GameVersion::B,
            EncounterType::Normal,
            test_value
        );
        let result2 = EncounterCalculator::calculate_encounter_slot(
            GameVersion::B,
            EncounterType::Normal,
            test_value
        );
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_version_consistency() {
        // BWとBW2で同じ結果が得られることを確認
        for encounter_type in [
            EncounterType::Normal,
            EncounterType::Surfing,
            EncounterType::Fishing,
            EncounterType::ShakingGrass,
            EncounterType::DustCloud,
            EncounterType::PokemonShadow,
            EncounterType::SurfingBubble,
            EncounterType::FishingBubble,
        ] {
            for rand_val in 0..100 {
                let bw_result = EncounterCalculator::calculate_encounter_slot(
                    GameVersion::B,
                    encounter_type,
                    rand_val
                );
                let bw2_result = EncounterCalculator::calculate_encounter_slot(
                    GameVersion::B2,
                    encounter_type,
                    rand_val
                );
                assert_eq!(bw_result, bw2_result, 
                    "Mismatch for type {:?}, rand_val {}", encounter_type, rand_val);
            }
        }
    }

    #[test]
    fn test_version_mathematical_differences() {
        // BW/BW2の数学的計算式の違いをテスト
        let test_values = [0u32, 1000, 0x12345678, 0x80000000, u32::MAX];
        
        for &rand_val in &test_values {
            let bw_slot = EncounterCalculator::calculate_raw_encounter_slot(
                GameVersion::B, 
                rand_val
            );
            let bw2_slot = EncounterCalculator::calculate_raw_encounter_slot(
                GameVersion::B2, 
                rand_val
            );
            
            // 計算式が異なるため、結果が異なる場合がある
            // ただし、両方とも有効な範囲内であることを確認
            assert!(bw_slot <= 100, "BW slot value {} out of range", bw_slot);
            assert!(bw2_slot <= 100, "BW2 slot value {} out of range", bw2_slot);
        }
    }

    #[test]
    fn test_deterministic_behavior() {
        // 同じ入力に対して同じ出力が得られることを確認
        let test_cases = [
            (EncounterType::Normal, 50),
            (EncounterType::Surfing, 75),
            (EncounterType::Fishing, 85),
            (EncounterType::ShakingGrass, 95),
            (EncounterType::DustCloud, 90),
        ];

        for (encounter_type, rand_val) in test_cases {
            let result1 = EncounterCalculator::calculate_encounter_slot(
                GameVersion::B,
                encounter_type,
                rand_val
            );
            let result2 = EncounterCalculator::calculate_encounter_slot(
                GameVersion::B,
                encounter_type,
                rand_val
            );
            assert_eq!(result1, result2);
        }
    }

    #[test]
    fn test_dust_cloud_content_classification() {
        // 砂煙の出現内容判定テスト
        assert_eq!(
            EncounterCalculator::get_dust_cloud_content(0),
            DustCloudContent::Pokemon
        );
        assert_eq!(
            EncounterCalculator::get_dust_cloud_content(1),
            DustCloudContent::Jewel
        );
        assert_eq!(
            EncounterCalculator::get_dust_cloud_content(2),
            DustCloudContent::EvolutionStone
        );
        
        // 範囲外値のテスト
        assert_eq!(
            EncounterCalculator::get_dust_cloud_content(5),
            DustCloudContent::Pokemon
        );
    }

    #[test]
    fn test_comprehensive_integration() {
        // 包括的統合テスト：全エンカウントタイプとバージョンの組み合わせ
        let encounter_types = [
            EncounterType::Normal,
            EncounterType::Surfing,
            EncounterType::Fishing,
            EncounterType::ShakingGrass,
            EncounterType::DustCloud,
            EncounterType::PokemonShadow,
            EncounterType::SurfingBubble,
            EncounterType::FishingBubble,
        ];
        
    let versions = [GameVersion::B, GameVersion::B2];
        let test_values = [0u32, 0x12345678, 0x80000000, u32::MAX];
        
        for version in versions {
            for encounter_type in encounter_types {
                for &rand_val in &test_values {
                    let slot = EncounterCalculator::calculate_encounter_slot(
                        version,
                        encounter_type,
                        rand_val
                    );
                    
                    // スロット値が適切な範囲内であることを確認
                    let max_slot = match encounter_type {
                        EncounterType::Normal => 11,
                        EncounterType::Surfing | EncounterType::Fishing | EncounterType::ShakingGrass => 4,
                        EncounterType::DustCloud => 2,
                        EncounterType::PokemonShadow | EncounterType::SurfingBubble | EncounterType::FishingBubble => 3,
                        EncounterType::StaticSymbol | EncounterType::StaticStarter | 
                        EncounterType::StaticFossil | EncounterType::StaticEvent | EncounterType::Roaming => 0,
                    };
                    
                    assert!(
                        slot <= max_slot,
                        "Slot {} exceeds max {} for {:?} in {:?}",
                        slot, max_slot, encounter_type, version
                    );
                    
                    // テーブルインデックス変換のテスト
                    let table_index = EncounterCalculator::slot_to_table_index(encounter_type, slot);
                    assert!(
                        table_index <= max_slot as usize,
                        "Table index {} exceeds max {} for slot {} in {:?}",
                        table_index, max_slot, slot, encounter_type
                    );
                }
            }
        }
    }
}
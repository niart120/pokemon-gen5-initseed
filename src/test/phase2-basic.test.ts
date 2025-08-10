/**
 * Simple Phase 2 validation test
 */

import { describe, it, expect } from 'vitest';
import { 
  getNatureName, 
  getShinyStatusName,
} from '../types/raw-pokemon-data';
import { determineGenderFromSpec } from '../lib/utils/gender-utils';
import { calculateLevel } from '../data/encounter-tables';
import { getGeneratedSpeciesById } from '../data/species/generated';

describe('Phase 2 Basic Validation', () => {
  describe('Raw Pokemon Data utilities', () => {
    it('should convert nature IDs correctly', () => {
      expect(getNatureName(0)).toBe('Hardy');
      expect(getNatureName(12)).toBe('Serious');
      expect(getNatureName(24)).toBe('Quirky');
    });

    it('should convert shiny types correctly', () => {
      expect(getShinyStatusName(0)).toBe('Normal');
      expect(getShinyStatusName(1)).toBe('Square Shiny');
      expect(getShinyStatusName(2)).toBe('Star Shiny');
    });

    it('should determine gender correctly (femaleThreshold semantics)', () => {
      // 50% female → threshold ~128
      expect(determineGenderFromSpec(100, { type: 'ratio', femaleThreshold: 128 })).toBe('Female');
      expect(determineGenderFromSpec(150, { type: 'ratio', femaleThreshold: 128 })).toBe('Male');
      // Genderless
      expect(determineGenderFromSpec(100, { type: 'genderless' })).toBe('Genderless');
    });
  });

  describe('Encounter Tables', () => {
    it('should calculate levels correctly', () => {
      expect(calculateLevel(0, { min: 5, max: 7 })).toBe(5);
      expect(calculateLevel(1, { min: 5, max: 7 })).toBe(6);
      expect(calculateLevel(2, { min: 5, max: 7 })).toBe(7);
      expect(calculateLevel(10, { min: 10, max: 10 })).toBe(10);
    });
  });

  describe('Pokemon Species (generated)', () => {
    it('should have generated species data for Gen 5 starters', () => {
      const snivy = getGeneratedSpeciesById(495);
      expect(snivy).toBeDefined();
      expect(snivy?.names.en).toBe('Snivy');

      const tepig = getGeneratedSpeciesById(498);
      expect(tepig).toBeDefined();
      expect(tepig?.names.en).toBe('Tepig');

      const oshawott = getGeneratedSpeciesById(501);
      expect(oshawott).toBeDefined();
      expect(oshawott?.names.en).toBe('Oshawott');
    });

    it('should return null for unknown species', () => {
      expect(getGeneratedSpeciesById(99999)).toBeNull();
    });
  });
});
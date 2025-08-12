import { describe, it, expect } from 'vitest';
import { pidHex, seedHex, natureName, shinyLabel } from '@/lib/utils/format-display';
import { DomainNatureNames, DomainShinyType } from '@/types/domain';

// Helper to build a bigint near 0xFFFFFFFFFFFFFFFF boundary (though we only display 16 digits)
const bigSeed = (v: number) => BigInt(v);

describe('format-display', () => {
  describe('pidHex', () => {
    it('formats 0 as 8 hex digits uppercase no prefix', () => {
      expect(pidHex(0)).toBe('00000000');
    });
    it('formats max 32-bit value', () => {
      expect(pidHex(0xFFFFFFFF)).toBe('FFFFFFFF');
    });
    it('truncates negative via >>> 0 behavior upstream (simulate)', () => {
      // Direct negative not passed normally; ensure large pattern not thrown
      expect(pidHex(-1 as unknown as number)).toBe('FFFFFFFF');
    });
  });

  describe('seedHex', () => {
    it('formats numeric seed to 16 digits', () => {
      expect(seedHex(0x1ABCDE)).toMatch(/^[0-9A-F]{16}$/); // zero-padded
    });
    it('formats bigint seed', () => {
      // 0x12345678 => 0000000012345678 (16桁固定 上位ゼロ埋め)
      expect(seedHex(bigSeed(0x12345678))).toBe('0000000012345678');
    });
  });

  describe('natureName', () => {
    it('maps valid ids within range', () => {
      expect(natureName(0)).toBe(DomainNatureNames[0]);
      expect(natureName(24)).toBe(DomainNatureNames[24]);
    });
    it('returns Unknown for out-of-range', () => {
      expect(natureName(-1)).toBe('Unknown');
      expect(natureName(999)).toBe('Unknown');
    });
  });

  describe('shinyLabel', () => {
    it('maps shiny types', () => {
      expect(shinyLabel(DomainShinyType.Normal)).toBe('No');
      expect(shinyLabel(DomainShinyType.Square)).toBe('Square');
      expect(shinyLabel(DomainShinyType.Star)).toBe('Star');
    });
    it('returns Unknown for unsupported', () => {
      expect(shinyLabel(9999)).toBe('Unknown');
    });
  });
});

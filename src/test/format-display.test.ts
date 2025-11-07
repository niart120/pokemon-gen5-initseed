import { describe, it, expect } from 'vitest';
import { pidHex, seedHex, natureName, shinyLabel, shinyDomainStatus, adaptGenerationResultDisplay } from '@/lib/utils/format-display';
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
      expect(natureName(0)).toBe(DomainNatureNames.en[0]);
      expect(natureName(24)).toBe(DomainNatureNames.en[24]);
      expect(natureName(0, 'ja')).toBe(DomainNatureNames.ja[0]);
    });
    it('returns Unknown for out-of-range', () => {
      expect(natureName(-1)).toBe('Unknown');
      expect(natureName(999)).toBe('Unknown');
      expect(natureName(999, 'ja')).toBe('不明');
    });
  });

  describe('shinyLabel', () => {
    it('maps shiny types', () => {
      expect(shinyLabel(DomainShinyType.Normal)).toBe('-');
      expect(shinyLabel(DomainShinyType.Square)).toBe('◇');
      expect(shinyLabel(DomainShinyType.Star)).toBe('☆');
      expect(shinyLabel(DomainShinyType.Star, 'ja')).toBe('☆');
    });
    it('returns Unknown for unsupported', () => {
      expect(shinyLabel(9999)).toBe('Unknown');
      expect(shinyLabel(9999, 'ja')).toBe('Unknown');
    });
  });

  describe('shinyDomainStatus', () => {
    it('maps shiny domain status values', () => {
      expect(shinyDomainStatus(DomainShinyType.Normal)).toBe('normal');
      expect(shinyDomainStatus(DomainShinyType.Square)).toBe('square');
      expect(shinyDomainStatus(DomainShinyType.Star)).toBe('star');
    });
    it('defaults unknown to normal', () => {
      expect(shinyDomainStatus(12345)).toBe('normal');
    });
  });

  describe('adaptGenerationResultDisplay', () => {
    it('adapts core fields', () => {
      const adapted = adaptGenerationResultDisplay({
        advance: 42,
        pid: 0x1234ABCD,
        nature: 5,
        shiny_type: DomainShinyType.Square,
        // unused fields for display omitted
      } as any);
      expect(adapted).toEqual({
        advance: 42,
        pidHex: '1234ABCD',
        natureName: DomainNatureNames.en[5],
        shinyLabel: '◇',
      });
    });
    it('handles out-of-range nature & unknown shiny', () => {
      const adapted = adaptGenerationResultDisplay({
        advance: 1,
        pid: 0,
        nature: 999,
        shiny_type: 999,
      } as any);
      expect(adapted.natureName).toBe('Unknown');
      expect(['-','◇','☆','Unknown']).toContain(adapted.shinyLabel);
    });
    it('respects locale argument', () => {
      const adapted = adaptGenerationResultDisplay({
        advance: 7,
        pid: 0xABCD,
        nature: 1,
        shiny_type: DomainShinyType.Star,
      } as any, 'ja');
      expect(adapted.natureName).toBe(DomainNatureNames.ja[1]);
      expect(adapted.shinyLabel).toBe('☆');
    });
  });
});

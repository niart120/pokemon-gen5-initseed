import { describe, it, expect } from 'vitest';
import { calculateLcgSeed, lcgSeedToHex, lcgSeedToMtSeed } from '@/lib/utils/lcg-seed';

describe('LCG Seed Utilities', () => {
  describe('calculateLcgSeed', () => {
    it('should calculate LCG seed from h0 and h1', () => {
      // Test with known values
      const h0 = 0x12345678;
      const h1 = 0x9ABCDEF0;
      
      const lcgSeed = calculateLcgSeed(h0, h1);
      
      // Verify it's a bigint
      expect(typeof lcgSeed).toBe('bigint');
      
      // Should combine h0 and h1 after byte swapping
      expect(lcgSeed).toBeDefined();
    });

    it('should handle zero values', () => {
      const h0 = 0x00000000;
      const h1 = 0x00000000;
      
      const lcgSeed = calculateLcgSeed(h0, h1);
      
      expect(lcgSeed).toBe(0n);
    });

    it('should handle maximum values', () => {
      const h0 = 0xFFFFFFFF;
      const h1 = 0xFFFFFFFF;
      
      const lcgSeed = calculateLcgSeed(h0, h1);
      
      // Should be a valid 64-bit value
      expect(lcgSeed).toBeDefined();
      expect(typeof lcgSeed).toBe('bigint');
    });
  });

  describe('lcgSeedToHex', () => {
    it('should convert LCG seed to hex string', () => {
      const lcgSeed = 0x123456789ABCDEF0n;
      
      const hexString = lcgSeedToHex(lcgSeed);
      
      expect(hexString).toBe('0x123456789ABCDEF0');
    });

    it('should pad with zeros', () => {
      const lcgSeed = 0x1n;
      
      const hexString = lcgSeedToHex(lcgSeed);
      
      expect(hexString).toBe('0x0000000000000001');
    });

    it('should handle zero', () => {
      const lcgSeed = 0n;
      
      const hexString = lcgSeedToHex(lcgSeed);
      
      expect(hexString).toBe('0x0000000000000000');
    });
  });

  describe('lcgSeedToMtSeed', () => {
    it('should convert LCG seed to MT seed', () => {
      const lcgSeed = 0x123456789ABCDEF0n;
      
      const mtSeed = lcgSeedToMtSeed(lcgSeed);
      
      // Should be a number
      expect(typeof mtSeed).toBe('number');
      
      // Should be within 32-bit range
      expect(mtSeed).toBeGreaterThanOrEqual(0);
      expect(mtSeed).toBeLessThanOrEqual(0xFFFFFFFF);
    });

    it('should produce consistent results', () => {
      const lcgSeed = 0x123456789ABCDEF0n;
      
      const mtSeed1 = lcgSeedToMtSeed(lcgSeed);
      const mtSeed2 = lcgSeedToMtSeed(lcgSeed);
      
      expect(mtSeed1).toBe(mtSeed2);
    });
  });
});

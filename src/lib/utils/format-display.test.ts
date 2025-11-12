import { describe, it, expect } from 'vitest';
import { calculateNeedleDirection, needleDirectionArrow, needleDisplay } from './format-display';

describe('Needle Direction Calculation', () => {
  it('should calculate needle direction correctly', () => {
    // Test case 1: seed = 0xE295B27C208D2A98 should give direction 7 (↖)
    const seed1 = 0xE295B27C208D2A98n;
    expect(calculateNeedleDirection(seed1)).toBe(7);
    
    // Test case 2: seed = 0x1AC6A030ADCBC4BBn should give direction 0 (↑)
    const seed2 = 0x1AC6A030ADCBC4BBn;
    expect(calculateNeedleDirection(seed2)).toBe(0);
    
    // Test case 3: seed = 0x8B3C1E8EE2F04F8An should give direction 4 (↓)
    const seed3 = 0x8B3C1E8EE2F04F8An;
    expect(calculateNeedleDirection(seed3)).toBe(4);
  });

  it('should map direction values to correct arrows', () => {
    expect(needleDirectionArrow(0)).toBe('↑');
    expect(needleDirectionArrow(1)).toBe('↗');
    expect(needleDirectionArrow(2)).toBe('→');
    expect(needleDirectionArrow(3)).toBe('↘');
    expect(needleDirectionArrow(4)).toBe('↓');
    expect(needleDirectionArrow(5)).toBe('↙');
    expect(needleDirectionArrow(6)).toBe('←');
    expect(needleDirectionArrow(7)).toBe('↖');
  });

  it('should format needle display correctly', () => {
    const seed = 0xE295B27C208D2A98n;
    const display = needleDisplay(seed);
    expect(display).toBe('↖(7)');
    
    const seed2 = 0x1AC6A030ADCBC4BBn;
    const display2 = needleDisplay(seed2);
    expect(display2).toBe('↑(0)');
  });

  it('should handle all 8 directions within range', () => {
    for (let dir = 0; dir < 8; dir++) {
      const arrow = needleDirectionArrow(dir);
      expect(arrow).not.toBe('?');
    }
  });
});

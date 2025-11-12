import { describe, it, expect } from 'vitest';
import { SEED_TEMPLATES } from '@/data/seed-templates';

describe('Seed Templates', () => {
  it('should have at least one template defined', () => {
    expect(SEED_TEMPLATES.length).toBeGreaterThan(0);
  });

  it('should have valid template structure', () => {
    SEED_TEMPLATES.forEach(template => {
      expect(template).toHaveProperty('name');
      expect(template).toHaveProperty('seeds');
      expect(typeof template.name).toBe('string');
      expect(Array.isArray(template.seeds)).toBe(true);
      expect(template.name.length).toBeGreaterThan(0);
      expect(template.seeds.length).toBeGreaterThan(0);
    });
  });

  it('should have valid seed values (32-bit integers)', () => {
    SEED_TEMPLATES.forEach(template => {
      template.seeds.forEach(seed => {
        expect(typeof seed).toBe('number');
        expect(seed).toBeGreaterThanOrEqual(0);
        expect(seed).toBeLessThanOrEqual(0xFFFFFFFF);
        expect(Number.isInteger(seed)).toBe(true);
      });
    });
  });

  it('should have unique template names', () => {
    const names = SEED_TEMPLATES.map(t => t.name);
    const uniqueNames = new Set(names);
    expect(uniqueNames.size).toBe(names.length);
  });

  it('should have specific expected templates', () => {
    const templateNames = SEED_TEMPLATES.map(t => t.name);
    
    // Check for expected templates
    expect(templateNames).toContain('BW 固定・野生 6V');
    expect(templateNames).toContain('BW2 固定・野生 5VA0');
  });
});

import { describe, it, expect } from 'vitest';
import { SEED_TEMPLATES, type TemplateVersion, type TemplateCategory } from '@/data/seed-templates';
import {
  getTemplateCategoryLabel,
  templateCategoryOptions,
  type TemplateCategoryFilter,
} from '@/lib/i18n/strings/search-template-selection';

describe('Seed Templates', () => {
  it('should have at least one template defined', () => {
    expect(SEED_TEMPLATES.length).toBeGreaterThan(0);
  });

  it('should have valid template structure', () => {
    SEED_TEMPLATES.forEach(template => {
      expect(template).toHaveProperty('name');
      expect(template).toHaveProperty('seeds');
      expect(template).toHaveProperty('version');
      expect(template).toHaveProperty('category');
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

  it('should have valid version values', () => {
    const validVersions: TemplateVersion[] = ['BW', 'BW2'];
    SEED_TEMPLATES.forEach(template => {
      expect(validVersions).toContain(template.version);
    });
  });

  it('should have valid category values', () => {
    const validCategories: TemplateCategory[] = ['stationary', 'roamer', 'egg'];
    SEED_TEMPLATES.forEach(template => {
      expect(validCategories).toContain(template.category);
    });
  });

  it('should have templates for both BW and BW2', () => {
    const bwTemplates = SEED_TEMPLATES.filter(t => t.version === 'BW');
    const bw2Templates = SEED_TEMPLATES.filter(t => t.version === 'BW2');
    
    expect(bwTemplates.length).toBeGreaterThan(0);
    expect(bw2Templates.length).toBeGreaterThan(0);
  });

  it('should have templates for all categories', () => {
    const stationaryTemplates = SEED_TEMPLATES.filter(t => t.category === 'stationary');
    const roamerTemplates = SEED_TEMPLATES.filter(t => t.category === 'roamer');
    const eggTemplates = SEED_TEMPLATES.filter(t => t.category === 'egg');
    
    expect(stationaryTemplates.length).toBeGreaterThan(0);
    expect(roamerTemplates.length).toBeGreaterThan(0);
    expect(eggTemplates.length).toBeGreaterThan(0);
  });

  it('should not contain test sample template', () => {
    const templateNames = SEED_TEMPLATES.map(t => t.name);
    expect(templateNames).not.toContain('テストサンプル');
  });
});

describe('Template Filtering', () => {
  it('should filter by version BW correctly', () => {
    const bwTemplates = SEED_TEMPLATES.filter(t => t.version === 'BW');
    
    bwTemplates.forEach(template => {
      expect(template.version).toBe('BW');
      expect(template.name).toMatch(/^BW\s/);
    });
  });

  it('should filter by version BW2 correctly', () => {
    const bw2Templates = SEED_TEMPLATES.filter(t => t.version === 'BW2');
    
    bw2Templates.forEach(template => {
      expect(template.version).toBe('BW2');
      expect(template.name).toMatch(/^BW2\s/);
    });
  });

  it('should filter by category stationary correctly', () => {
    const stationaryTemplates = SEED_TEMPLATES.filter(t => t.category === 'stationary');
    
    stationaryTemplates.forEach(template => {
      expect(template.category).toBe('stationary');
      expect(template.name).toContain('固定・野生');
    });
  });

  it('should filter by category roamer correctly', () => {
    const roamerTemplates = SEED_TEMPLATES.filter(t => t.category === 'roamer');
    
    roamerTemplates.forEach(template => {
      expect(template.category).toBe('roamer');
      expect(template.name).toContain('徘徊');
    });
  });

  it('should filter by category egg correctly', () => {
    const eggTemplates = SEED_TEMPLATES.filter(t => t.category === 'egg');
    
    eggTemplates.forEach(template => {
      expect(template.category).toBe('egg');
      expect(template.name).toContain('孵化');
    });
  });

  it('should filter by version and category combined', () => {
    const bwStationaryTemplates = SEED_TEMPLATES.filter(
      t => t.version === 'BW' && t.category === 'stationary'
    );
    
    expect(bwStationaryTemplates.length).toBeGreaterThan(0);
    bwStationaryTemplates.forEach(template => {
      expect(template.version).toBe('BW');
      expect(template.category).toBe('stationary');
    });
  });

  it('should return all templates when no filter applied', () => {
    const allTemplates = SEED_TEMPLATES.filter(() => true);
    expect(allTemplates.length).toBe(SEED_TEMPLATES.length);
  });
});

describe('Template Category i18n', () => {
  it('should have all category options defined', () => {
    expect(templateCategoryOptions).toContain('all');
    expect(templateCategoryOptions).toContain('stationary');
    expect(templateCategoryOptions).toContain('roamer');
    expect(templateCategoryOptions).toContain('egg');
    expect(templateCategoryOptions.length).toBe(4);
  });

  it('should return Japanese labels for all categories', () => {
    const categories: TemplateCategoryFilter[] = ['all', 'stationary', 'roamer', 'egg'];
    
    categories.forEach(category => {
      const label = getTemplateCategoryLabel(category, 'ja');
      expect(typeof label).toBe('string');
      expect(label.length).toBeGreaterThan(0);
    });

    expect(getTemplateCategoryLabel('all', 'ja')).toBe('全て');
    expect(getTemplateCategoryLabel('stationary', 'ja')).toBe('固定・野生');
    expect(getTemplateCategoryLabel('roamer', 'ja')).toBe('徘徊');
    expect(getTemplateCategoryLabel('egg', 'ja')).toBe('孵化');
  });

  it('should return English labels for all categories', () => {
    const categories: TemplateCategoryFilter[] = ['all', 'stationary', 'roamer', 'egg'];
    
    categories.forEach(category => {
      const label = getTemplateCategoryLabel(category, 'en');
      expect(typeof label).toBe('string');
      expect(label.length).toBeGreaterThan(0);
    });

    expect(getTemplateCategoryLabel('all', 'en')).toBe('All');
    expect(getTemplateCategoryLabel('stationary', 'en')).toBe('Stationary/Wild');
    expect(getTemplateCategoryLabel('roamer', 'en')).toBe('Roamer');
    expect(getTemplateCategoryLabel('egg', 'en')).toBe('Egg');
  });
});

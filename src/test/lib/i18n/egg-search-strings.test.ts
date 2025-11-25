import { describe, it, expect } from 'vitest';
import {
  eggSearchPanelTitle,
  eggSearchRunCardTitle,
  eggSearchStatusPrefix,
  eggSearchFoundLabel,
  eggSearchElapsedLabel,
  eggSearchProgressLabel,
  eggSearchButtonLabels,
  eggSearchStatusLabels,
  eggSearchParamsCardTitle,
  eggSearchParamsLabels,
  eggSearchFilterCardTitle,
  eggSearchFilterLabels,
  eggSearchResultsCardTitle,
  eggSearchResultsEmpty,
  eggSearchResultsCountLabel,
  eggSearchResultsTableHeaders,
  getEggSearchStatusLabel,
  formatEggSearchElapsed,
  formatEggSearchResultsCount,
} from '@/lib/i18n/strings/egg-search';

describe('egg-search i18n strings', () => {
  describe('locale texts', () => {
    it('should have both ja and en for panel title', () => {
      expect(eggSearchPanelTitle.ja).toBeDefined();
      expect(eggSearchPanelTitle.en).toBeDefined();
    });

    it('should have both ja and en for run card title', () => {
      expect(eggSearchRunCardTitle.ja).toBe('検索制御');
      expect(eggSearchRunCardTitle.en).toBe('Search Control');
    });

    it('should have both ja and en for status prefix', () => {
      expect(eggSearchStatusPrefix.ja).toBe('ステータス');
      expect(eggSearchStatusPrefix.en).toBe('Status');
    });

    it('should have both ja and en for found label', () => {
      expect(eggSearchFoundLabel.ja).toBe('発見数');
      expect(eggSearchFoundLabel.en).toBe('Found');
    });

    it('should have both ja and en for elapsed label', () => {
      expect(eggSearchElapsedLabel.ja).toBe('経過時間');
      expect(eggSearchElapsedLabel.en).toBe('Elapsed');
    });

    it('should have both ja and en for progress label', () => {
      expect(eggSearchProgressLabel.ja).toBe('進捗');
      expect(eggSearchProgressLabel.en).toBe('Progress');
    });
  });

  describe('button labels', () => {
    it('should have start button labels', () => {
      expect(eggSearchButtonLabels.start.ja).toBe('検索開始');
      expect(eggSearchButtonLabels.start.en).toBe('Start Search');
    });

    it('should have stop button labels', () => {
      expect(eggSearchButtonLabels.stop.ja).toBe('停止');
      expect(eggSearchButtonLabels.stop.en).toBe('Stop');
    });

    it('should have stopping button labels', () => {
      expect(eggSearchButtonLabels.stopping.ja).toBe('停止中...');
      expect(eggSearchButtonLabels.stopping.en).toBe('Stopping...');
    });
  });

  describe('status labels', () => {
    it('should have all status labels for ja', () => {
      expect(eggSearchStatusLabels.ja.idle).toBe('アイドル');
      expect(eggSearchStatusLabels.ja.starting).toBe('開始中');
      expect(eggSearchStatusLabels.ja.running).toBe('検索中');
      expect(eggSearchStatusLabels.ja.stopping).toBe('停止中');
      expect(eggSearchStatusLabels.ja.completed).toBe('完了');
      expect(eggSearchStatusLabels.ja.error).toBe('エラー');
    });

    it('should have all status labels for en', () => {
      expect(eggSearchStatusLabels.en.idle).toBe('Idle');
      expect(eggSearchStatusLabels.en.starting).toBe('Starting');
      expect(eggSearchStatusLabels.en.running).toBe('Searching');
      expect(eggSearchStatusLabels.en.stopping).toBe('Stopping');
      expect(eggSearchStatusLabels.en.completed).toBe('Completed');
      expect(eggSearchStatusLabels.en.error).toBe('Error');
    });
  });

  describe('params card labels', () => {
    it('should have params card title', () => {
      expect(eggSearchParamsCardTitle.ja).toBe('検索条件');
      expect(eggSearchParamsCardTitle.en).toBe('Search Parameters');
    });

    it('should have all params labels', () => {
      expect(eggSearchParamsLabels.startDate.ja).toBeDefined();
      expect(eggSearchParamsLabels.startDate.en).toBeDefined();
      expect(eggSearchParamsLabels.endDate.ja).toBeDefined();
      expect(eggSearchParamsLabels.endDate.en).toBeDefined();
      expect(eggSearchParamsLabels.timeRange.ja).toBeDefined();
      expect(eggSearchParamsLabels.timeRange.en).toBeDefined();
      expect(eggSearchParamsLabels.userOffset.ja).toBeDefined();
      expect(eggSearchParamsLabels.userOffset.en).toBeDefined();
      expect(eggSearchParamsLabels.advanceCount.ja).toBeDefined();
      expect(eggSearchParamsLabels.advanceCount.en).toBeDefined();
      expect(eggSearchParamsLabels.keyInput.ja).toBeDefined();
      expect(eggSearchParamsLabels.keyInput.en).toBeDefined();
      expect(eggSearchParamsLabels.maleParentIv.ja).toBeDefined();
      expect(eggSearchParamsLabels.maleParentIv.en).toBeDefined();
      expect(eggSearchParamsLabels.femaleParentIv.ja).toBeDefined();
      expect(eggSearchParamsLabels.femaleParentIv.en).toBeDefined();
    });
  });

  describe('filter card labels', () => {
    it('should have filter card title', () => {
      expect(eggSearchFilterCardTitle.ja).toBe('フィルター');
      expect(eggSearchFilterCardTitle.en).toBe('Filter');
    });

    it('should have shiny filter labels', () => {
      expect(eggSearchFilterLabels.shinyOnly.ja).toBe('色違いのみ');
      expect(eggSearchFilterLabels.shinyOnly.en).toBe('Shiny Only');
      expect(eggSearchFilterLabels.shinyHint.ja).toBeDefined();
      expect(eggSearchFilterLabels.shinyHint.en).toBeDefined();
    });
  });

  describe('results card labels', () => {
    it('should have results card title', () => {
      expect(eggSearchResultsCardTitle.ja).toBe('検索結果');
      expect(eggSearchResultsCardTitle.en).toBe('Search Results');
    });

    it('should have empty results message', () => {
      expect(eggSearchResultsEmpty.ja).toBe('結果がありません');
      expect(eggSearchResultsEmpty.en).toBe('No results');
    });

    it('should have results count label', () => {
      expect(eggSearchResultsCountLabel.ja).toBe('件');
      expect(eggSearchResultsCountLabel.en).toBe('results');
    });

    it('should have all table headers', () => {
      expect(eggSearchResultsTableHeaders.bootTime.ja).toBeDefined();
      expect(eggSearchResultsTableHeaders.bootTime.en).toBeDefined();
      expect(eggSearchResultsTableHeaders.timer0.ja).toBeDefined();
      expect(eggSearchResultsTableHeaders.timer0.en).toBeDefined();
      expect(eggSearchResultsTableHeaders.vcount.ja).toBeDefined();
      expect(eggSearchResultsTableHeaders.vcount.en).toBeDefined();
      expect(eggSearchResultsTableHeaders.lcgSeed.ja).toBeDefined();
      expect(eggSearchResultsTableHeaders.lcgSeed.en).toBeDefined();
      expect(eggSearchResultsTableHeaders.advance.ja).toBeDefined();
      expect(eggSearchResultsTableHeaders.advance.en).toBeDefined();
      expect(eggSearchResultsTableHeaders.nature.ja).toBeDefined();
      expect(eggSearchResultsTableHeaders.nature.en).toBeDefined();
      expect(eggSearchResultsTableHeaders.ivs.ja).toBeDefined();
      expect(eggSearchResultsTableHeaders.ivs.en).toBeDefined();
      expect(eggSearchResultsTableHeaders.shiny.ja).toBeDefined();
      expect(eggSearchResultsTableHeaders.shiny.en).toBeDefined();
      expect(eggSearchResultsTableHeaders.stable.ja).toBeDefined();
      expect(eggSearchResultsTableHeaders.stable.en).toBeDefined();
    });
  });

  describe('getEggSearchStatusLabel', () => {
    it('should return correct status label for ja', () => {
      expect(getEggSearchStatusLabel('idle', 'ja')).toBe('アイドル');
      expect(getEggSearchStatusLabel('running', 'ja')).toBe('検索中');
      expect(getEggSearchStatusLabel('completed', 'ja')).toBe('完了');
    });

    it('should return correct status label for en', () => {
      expect(getEggSearchStatusLabel('idle', 'en')).toBe('Idle');
      expect(getEggSearchStatusLabel('running', 'en')).toBe('Searching');
      expect(getEggSearchStatusLabel('completed', 'en')).toBe('Completed');
    });
  });

  describe('formatEggSearchElapsed', () => {
    it('should format seconds for ja', () => {
      expect(formatEggSearchElapsed(5000, 'ja')).toBe('5秒');
      expect(formatEggSearchElapsed(30000, 'ja')).toBe('30秒');
    });

    it('should format seconds for en', () => {
      expect(formatEggSearchElapsed(5000, 'en')).toBe('5s');
      expect(formatEggSearchElapsed(30000, 'en')).toBe('30s');
    });

    it('should format minutes and seconds for ja', () => {
      expect(formatEggSearchElapsed(90000, 'ja')).toBe('1分30秒');
      expect(formatEggSearchElapsed(125000, 'ja')).toBe('2分5秒');
    });

    it('should format minutes and seconds for en', () => {
      expect(formatEggSearchElapsed(90000, 'en')).toBe('1m 30s');
      expect(formatEggSearchElapsed(125000, 'en')).toBe('2m 5s');
    });
  });

  describe('formatEggSearchResultsCount', () => {
    it('should format count for ja without space before counter', () => {
      expect(formatEggSearchResultsCount(100, 'ja')).toBe('100件');
      expect(formatEggSearchResultsCount(1000, 'ja')).toBe('1,000件');
    });

    it('should format count for en with space before word', () => {
      expect(formatEggSearchResultsCount(100, 'en')).toBe('100 results');
      expect(formatEggSearchResultsCount(1000, 'en')).toBe('1,000 results');
    });

    it('should handle zero', () => {
      expect(formatEggSearchResultsCount(0, 'ja')).toBe('0件');
      expect(formatEggSearchResultsCount(0, 'en')).toBe('0 results');
    });
  });
});

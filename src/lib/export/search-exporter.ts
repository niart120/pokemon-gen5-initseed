/**
 * Export functionality for seed search results
 * Supports CSV, JSON, and text formats
 */

import type { SearchResult } from '@/types/search';
import { formatKeyInputForDisplay, KEY_INPUT_DISPLAY_FALLBACK } from '@/lib/utils/key-input';
import {
  formatBootTimestampDisplay,
  formatTimer0Hex,
  formatVCountHex,
} from '@/lib/generation/result-formatters';
import {
  downloadFile,
  copyToClipboard,
  generateFilename,
  type ExportFormat,
} from './file-utils';

export interface SearchExportOptions {
  format: ExportFormat;
  includeDetails?: boolean;
  includeMessage?: boolean;
  includeHash?: boolean;
}

/**
 * Export search results in the specified format
 */
export function exportSearchResults(
  results: SearchResult[],
  options: SearchExportOptions,
  locale: 'ja' | 'en'
): string {
  switch (options.format) {
    case 'csv':
      return exportToCSV(results, options, locale);
    case 'json':
      return exportToJSON(results, options, locale);
    case 'txt':
      return exportToText(results, options, locale);
    default:
      throw new Error(`Unsupported export format: ${options.format}`);
  }
}

/**
 * Export to CSV format
 */
function exportToCSV(
  results: SearchResult[],
  options: SearchExportOptions,
  locale: 'ja' | 'en'
): string {
  const headers = [
    'Seed (Hex)',
    'DateTime',
    'Timer0 (Hex)',
    'VCount (Hex)',
    'ROM',
    'Region',
    'Hardware',
  ];

  if (options.includeDetails) {
    headers.push('MAC Address', 'Key Input');
  }

  if (options.includeMessage) {
    headers.push('Message');
  }

  if (options.includeHash) {
    headers.push('SHA1 Hash');
  }

  const csvLines = [headers.join(',')];

  results.forEach((result) => {
    const timestampDisplay = formatBootTimestampDisplay(result.datetime, locale);
    const timer0Display = formatTimer0Hex(result.timer0);
    const vcountDisplay = formatVCountHex(result.vcount);
    const row = [
      `0x${result.seed.toString(16).padStart(8, '0')}`,
      timestampDisplay,
      timer0Display,
      vcountDisplay,
      result.romVersion,
      result.romRegion,
      result.hardware,
    ];

    if (options.includeDetails) {
      const macAddress =
        result.macAddress?.map((b) => b.toString(16).padStart(2, '0')).join(':') || '';
      const keyInput = formatKeyInputForDisplay(result.keyCode, undefined);
      row.push(macAddress, keyInput);
    }

    if (options.includeMessage) {
      const message =
        result.message?.map((m) => `0x${m.toString(16).padStart(8, '0')}`).join(' ') || '';
      row.push(`"${message}"`);
    }

    if (options.includeHash) {
      row.push(result.hash || '');
    }

    csvLines.push(row.join(','));
  });

  return csvLines.join('\n');
}

/**
 * Export to JSON format
 */
function exportToJSON(
  results: SearchResult[],
  options: SearchExportOptions,
  locale: 'ja' | 'en'
): string {
  const exportData = {
    exportDate: new Date().toISOString(),
    totalResults: results.length,
    format: 'json',
    results: results.map((result) => {
      const exportResult: {
        seed: string;
        seedDecimal: number;
        dateTime: string;
        timer0: string;
        timer0Decimal: number;
        vcount: string;
        vcountDecimal: number;
        rom: {
          version: string;
          region: string;
          hardware: string;
        };
        macAddress?: string[];
        keyInput?: string | null;
        message?: string[];
        sha1Hash?: string;
      } = {
        seed: `0x${result.seed.toString(16).padStart(8, '0')}`,
        seedDecimal: result.seed,
        dateTime: formatBootTimestampDisplay(result.datetime, locale),
        timer0: formatTimer0Hex(result.timer0),
        timer0Decimal: result.timer0,
        vcount: formatVCountHex(result.vcount),
        vcountDecimal: result.vcount,
        rom: {
          version: result.romVersion,
          region: result.romRegion,
          hardware: result.hardware,
        },
      };

      if (options.includeDetails) {
        exportResult.macAddress = result.macAddress?.map(
          (b) => `0x${b.toString(16).padStart(2, '0')}`
        );
        const formattedKey = formatKeyInputForDisplay(result.keyCode, undefined);
        exportResult.keyInput = formattedKey !== KEY_INPUT_DISPLAY_FALLBACK ? formattedKey : null;
      }

      if (options.includeMessage) {
        exportResult.message = result.message?.map((m) => `0x${m.toString(16).padStart(8, '0')}`);
      }

      if (options.includeHash) {
        exportResult.sha1Hash = result.hash;
      }

      return exportResult;
    }),
  };

  return JSON.stringify(exportData, null, 2);
}

/**
 * Export to text format
 */
function exportToText(
  results: SearchResult[],
  options: SearchExportOptions,
  locale: 'ja' | 'en'
): string {
  const lines = [
    'Pokemon BW/BW2 Initial Seed Search Results',
    `Export Date: ${new Date().toISOString()}`,
    `Total Results: ${results.length}`,
    '',
    '================================================================',
  ];

  results.forEach((result, index) => {
    lines.push(`Result #${index + 1}:`);
    lines.push(`  Seed: 0x${result.seed.toString(16).padStart(8, '0')} (${result.seed})`);
    lines.push(`  DateTime: ${formatBootTimestampDisplay(result.datetime, locale)}`);
    lines.push(`  Timer0: ${formatTimer0Hex(result.timer0)} (${result.timer0})`);
    lines.push(`  VCount: ${formatVCountHex(result.vcount)} (${result.vcount})`);
    lines.push(`  ROM: ${result.romVersion} ${result.romRegion} (${result.hardware})`);

    if (options.includeDetails) {
      const macAddress =
        result.macAddress?.map((b) => b.toString(16).padStart(2, '0')).join(':') || KEY_INPUT_DISPLAY_FALLBACK;
      const keyInput = formatKeyInputForDisplay(result.keyCode, undefined);
      lines.push(`  MAC Address: ${macAddress}`);
      lines.push(`  Key Input: ${keyInput}`);
    }

    if (options.includeHash) {
      lines.push(`  SHA1 Hash: ${result.hash || KEY_INPUT_DISPLAY_FALLBACK}`);
    }

    if (options.includeMessage) {
      const message =
        result.message?.map((m) => `0x${m.toString(16).padStart(8, '0')}`).join(' ') || KEY_INPUT_DISPLAY_FALLBACK;
      lines.push(`  Message: ${message}`);
    }

    lines.push('');
  });

  return lines.join('\n');
}

// Re-export file utilities for backward compatibility
export { downloadFile, copyToClipboard, generateFilename };

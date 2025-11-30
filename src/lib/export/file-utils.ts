/**
 * File export utility functions
 * Shared across all exporters (Search, Generation, Egg, etc.)
 */

export type ExportFormat = 'csv' | 'json' | 'txt';

/**
 * MIME types for export formats
 */
export const MIME_TYPES: Record<ExportFormat, string> = {
  csv: 'text/csv',
  json: 'application/json',
  txt: 'text/plain',
};

/**
 * Download content as a file
 */
export function downloadFile(
  content: string,
  filename: string,
  mimeType: string = 'text/plain'
): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);

  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.style.display = 'none';

  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  URL.revokeObjectURL(url);
}

/**
 * Copy content to clipboard
 * @returns true if successful, false otherwise
 */
export async function copyToClipboard(content: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(content);
    return true;
  } catch (error) {
    console.error('Failed to copy to clipboard:', error);
    return false;
  }
}

/**
 * Generate filename with timestamp
 * @param format - Export format extension
 * @param prefix - Filename prefix (default: 'export')
 * @returns Filename with format: {prefix}-{YYYYMMDD}-{HHMMSS}.{format}
 */
export function generateFilename(
  format: ExportFormat,
  prefix: string = 'export'
): string {
  const now = new Date();
  const dateStr = now.toISOString().split('T')[0].replace(/-/g, '');
  const timeStr = now.toTimeString().split(' ')[0].replace(/:/g, '');
  return `${prefix}-${dateStr}-${timeStr}.${format}`;
}

/**
 * Export and download content with appropriate MIME type
 */
export function exportAndDownload(
  content: string,
  format: ExportFormat,
  filenamePrefix: string = 'export'
): void {
  const filename = generateFilename(format, filenamePrefix);
  const mimeType = MIME_TYPES[format];
  downloadFile(content, filename, mimeType);
}

/**
 * Export and copy to clipboard
 * @returns true if successful, false otherwise
 */
export async function exportAndCopy(content: string): Promise<boolean> {
  return copyToClipboard(content);
}

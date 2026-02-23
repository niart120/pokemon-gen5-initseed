/**
 * EggSearchExportButton
 * タマゴ起動時間検索結果のエクスポートボタンコンポーネント
 */

import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Download, Copy, Check } from '@phosphor-icons/react';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import {
  eggSearchExportDialogTitle,
  eggSearchExportFormatLabel,
  eggSearchExportFormatOptions,
  eggSearchExportDownloadLabel,
  formatEggSearchExportTriggerLabel,
  formatEggSearchExportSummary,
  formatEggSearchExportCopyLabel,
} from '@/lib/i18n/strings/egg-search-export';
import {
  exportEggSearchResults,
  type EggSearchExportOptions,
} from '@/lib/export/egg-search-exporter';
import {
  downloadFile,
  copyToClipboard,
  generateFilename,
  MIME_TYPES,
  type ExportFormat,
} from '@/lib/export/file-utils';
import type { EggBootTimingSearchResult } from '@/types/egg-boot-timing-search';

interface EggSearchExportButtonProps {
  results: EggBootTimingSearchResult[];
  disabled?: boolean;
}

export function EggSearchExportButton({
  results,
  disabled = false,
}: EggSearchExportButtonProps) {
  const locale = useLocale();
  const [isOpen, setIsOpen] = useState(false);
  const [format, setFormat] = useState<ExportFormat>('csv');
  const [copied, setCopied] = useState(false);

  const rowCount = results.length;
  const triggerLabel = formatEggSearchExportTriggerLabel(rowCount, locale);
  const dialogTitle = resolveLocaleValue(eggSearchExportDialogTitle, locale);
  const formatLabel = resolveLocaleValue(eggSearchExportFormatLabel, locale);
  const formatLabels = {
    csv: resolveLocaleValue(eggSearchExportFormatOptions.csv, locale),
    json: resolveLocaleValue(eggSearchExportFormatOptions.json, locale),
    txt: resolveLocaleValue(eggSearchExportFormatOptions.txt, locale),
  };
  const downloadLabel = resolveLocaleValue(eggSearchExportDownloadLabel, locale);
  const copyLabel = formatEggSearchExportCopyLabel(copied, locale);
  const summaryText = formatEggSearchExportSummary(rowCount, locale);

  const handleExport = async (download: boolean) => {
    try {
      const options: EggSearchExportOptions = {
        format,
      };

      const content = exportEggSearchResults(results, options, {
        locale,
      });

      if (download) {
        const filename = generateFilename(format, 'egg-search-results');
        downloadFile(content, filename, MIME_TYPES[format]);
      } else {
        const success = await copyToClipboard(content);
        if (success) {
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
        }
      }

      setIsOpen(false);
    } catch (error) {
      console.error('Egg search export failed:', error);
    }
  };

  return (
    <Dialog
      open={isOpen}
      onOpenChange={(open) => {
        setIsOpen(open);
        if (!open) {
          setCopied(false);
        }
      }}
    >
      <DialogTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          disabled={disabled || rowCount === 0}
          className="gap-2"
        >
          <Download size={16} />
          {triggerLabel}
        </Button>
      </DialogTrigger>

      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{dialogTitle}</DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          <div>
            <Label htmlFor="egg-search-export-format">{formatLabel}</Label>
            <Select
              value={format}
              onValueChange={(value) => setFormat(value as ExportFormat)}
            >
              <SelectTrigger id="egg-search-export-format">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="csv">{formatLabels.csv}</SelectItem>
                <SelectItem value="json">{formatLabels.json}</SelectItem>
                <SelectItem value="txt">{formatLabels.txt}</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="text-sm text-muted-foreground">{summaryText}</div>

          <div className="flex gap-2">
            <Button onClick={() => handleExport(true)} className="flex-1 gap-2">
              <Download size={16} />
              {downloadLabel}
            </Button>
            <Button
              variant="outline"
              onClick={() => handleExport(false)}
              className="flex-1 gap-2"
            >
              {copied ? <Check size={16} /> : <Copy size={16} />}
              {copyLabel}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

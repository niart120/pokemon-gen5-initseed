/**
 * EggExportButton
 * タマゴ生成結果のエクスポートボタンコンポーネント
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
import { Checkbox } from '@/components/ui/checkbox';
import { Download, Copy, Check } from '@phosphor-icons/react';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import {
  eggExportDialogTitle,
  eggExportFormatLabel,
  eggExportFormatOptions,
  eggExportDownloadLabel,
  eggExportAdditionalDataLabel,
  eggExportIncludeBootTimingLabel,
  formatEggExportTriggerLabel,
  formatEggExportSummary,
  formatEggExportCopyLabel,
} from '@/lib/i18n/strings/egg-export';
import {
  exportEggGenerationResults,
  type EggGenerationExportOptions,
} from '@/lib/export/egg-generation-exporter';
import {
  downloadFile,
  copyToClipboard,
  generateFilename,
  MIME_TYPES,
  type ExportFormat,
} from '@/lib/export/file-utils';
import type { EnumeratedEggDataWithBootTiming } from '@/types/egg';

interface EggExportButtonProps {
  results: EnumeratedEggDataWithBootTiming[];
  isBootTimingMode: boolean;
  disabled?: boolean;
}

export function EggExportButton({
  results,
  isBootTimingMode,
  disabled = false,
}: EggExportButtonProps) {
  const locale = useLocale();
  const [isOpen, setIsOpen] = useState(false);
  const [format, setFormat] = useState<ExportFormat>('csv');
  const [includeBootTiming, setIncludeBootTiming] = useState(false);
  const [copied, setCopied] = useState(false);

  const rowCount = results.length;
  const triggerLabel = formatEggExportTriggerLabel(rowCount, locale);
  const dialogTitle = resolveLocaleValue(eggExportDialogTitle, locale);
  const formatLabel = resolveLocaleValue(eggExportFormatLabel, locale);
  const formatLabels = {
    csv: resolveLocaleValue(eggExportFormatOptions.csv, locale),
    json: resolveLocaleValue(eggExportFormatOptions.json, locale),
    txt: resolveLocaleValue(eggExportFormatOptions.txt, locale),
  };
  const downloadLabel = resolveLocaleValue(eggExportDownloadLabel, locale);
  const copyLabel = formatEggExportCopyLabel(copied, locale);
  const additionalDataLabel = resolveLocaleValue(eggExportAdditionalDataLabel, locale);
  const includeBootTimingLabel = resolveLocaleValue(eggExportIncludeBootTimingLabel, locale);
  const summaryText = formatEggExportSummary(rowCount, locale);

  const handleExport = async (download: boolean) => {
    try {
      const options: EggGenerationExportOptions = {
        format,
        includeBootTiming: includeBootTiming && isBootTimingMode,
      };

      const content = exportEggGenerationResults(results, options, {
        locale,
        isBootTimingMode,
      });

      if (download) {
        const filename = generateFilename(format, 'egg-results');
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
      console.error('Egg export failed:', error);
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
            <Label htmlFor="egg-export-format">{formatLabel}</Label>
            <Select
              value={format}
              onValueChange={(value) => setFormat(value as ExportFormat)}
            >
              <SelectTrigger id="egg-export-format">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="csv">{formatLabels.csv}</SelectItem>
                <SelectItem value="json">{formatLabels.json}</SelectItem>
                <SelectItem value="txt">{formatLabels.txt}</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {isBootTimingMode && (
            <div className="space-y-3">
              <Label>{additionalDataLabel}</Label>
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="egg-include-boot-timing"
                  checked={includeBootTiming}
                  onCheckedChange={(checked) => setIncludeBootTiming(Boolean(checked))}
                />
                <Label htmlFor="egg-include-boot-timing" className="text-sm font-normal">
                  {includeBootTimingLabel}
                </Label>
              </div>
            </div>
          )}

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

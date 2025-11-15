import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Download, Copy, Check } from '@phosphor-icons/react';
import { exportGenerationResults } from '@/lib/export/generation-exporter';
import type { GenerationResult } from '@/types/generation';
import { useLocale } from '@/lib/i18n/locale-context';
import {
  formatGenerationExportSummary,
  formatGenerationExportTriggerLabel,
  generationExportCopiedLabel,
  generationExportCopyLabel,
  generationExportDialogTitle,
  generationExportDownloadLabel,
  generationExportFormatLabel,
  generationExportFormatOptions,
} from '@/lib/i18n/strings/generation-export';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';

type ExportFormat = 'csv' | 'json' | 'txt';

interface GenerationExportButtonProps {
  results: GenerationResult[];
  disabled?: boolean;
}

export function GenerationExportButton({ results, disabled = false }: GenerationExportButtonProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [format, setFormat] = useState<ExportFormat>('csv');
  const [copied, setCopied] = useState(false);
  const locale = useLocale();
  const triggerLabel = formatGenerationExportTriggerLabel(results.length, locale);
  const dialogTitle = resolveLocaleValue(generationExportDialogTitle, locale);
  const formatLabel = resolveLocaleValue(generationExportFormatLabel, locale);
  const formatLabels = {
    csv: resolveLocaleValue(generationExportFormatOptions.csv, locale),
    json: resolveLocaleValue(generationExportFormatOptions.json, locale),
    txt: resolveLocaleValue(generationExportFormatOptions.txt, locale),
  };
  const downloadLabel = resolveLocaleValue(generationExportDownloadLabel, locale);
  const copyLabel = resolveLocaleValue(generationExportCopyLabel, locale);
  const copiedLabel = resolveLocaleValue(generationExportCopiedLabel, locale);
  const summaryText = formatGenerationExportSummary(results.length, locale);

  const handleExport = async (download: boolean) => {
    try {
      const content = exportGenerationResults(results, { format });
      if (download) {
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `generation-results.${format}`;
        link.click();
        URL.revokeObjectURL(url);
      } else if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(content);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      }
      setIsOpen(false);
    } catch (error) {
      console.error('Generation export failed:', error);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          disabled={disabled || results.length === 0}
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
            <Label htmlFor="generation-export-format">{formatLabel}</Label>
            <Select value={format} onValueChange={value => setFormat(value as ExportFormat)}>
              <SelectTrigger id="generation-export-format">
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
              {copied ? copiedLabel : copyLabel}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

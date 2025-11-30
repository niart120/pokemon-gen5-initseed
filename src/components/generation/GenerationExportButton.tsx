import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import { Download, Copy, Check } from '@phosphor-icons/react';
import { exportGenerationResults } from '@/lib/export/generation-exporter';
import { downloadFile, copyToClipboard, generateFilename, MIME_TYPES } from '@/lib/export/file-utils';
import type { EncounterTable } from '@/data/encounter-tables';
import type { GenderRatio } from '@/types/pokemon-raw';
import type { GenerationParams, GenerationResult } from '@/types/generation';
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
  generationExportAdditionalDataLabel,
  generationExportIncludeAdvancedLabel,
} from '@/lib/i18n/strings/generation-export';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';

type ExportFormat = 'csv' | 'json' | 'txt';

interface GenerationExportButtonProps {
  rows: GenerationResult[];
  encounterTable?: EncounterTable;
  genderRatios?: Map<number, GenderRatio>;
  abilityCatalog?: Map<number, string[]>;
  version: GenerationParams['version'];
  baseSeed?: bigint;
  disabled?: boolean;
}

export function GenerationExportButton({
  rows,
  encounterTable,
  genderRatios,
  abilityCatalog,
  version,
  baseSeed,
  disabled = false,
}: GenerationExportButtonProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [format, setFormat] = useState<ExportFormat>('csv');
  const [includeAdvanced, setIncludeAdvanced] = useState(false);
  const [copied, setCopied] = useState(false);
  const locale = useLocale();
  const rowCount = rows.length;
  const triggerLabel = formatGenerationExportTriggerLabel(rowCount, locale);
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
  const additionalDataLabel = resolveLocaleValue(generationExportAdditionalDataLabel, locale);
  const includeAdvancedLabel = resolveLocaleValue(generationExportIncludeAdvancedLabel, locale);
  const summaryText = formatGenerationExportSummary(rowCount, locale);

  const handleExport = async (download: boolean) => {
    try {
      const content = exportGenerationResults(rows, { format, includeAdvancedFields: includeAdvanced }, {
        encounterTable,
        genderRatios,
        abilityCatalog,
        locale,
        version,
        baseSeed,
      });
      if (download) {
        const filename = generateFilename(format, 'generation-results');
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
      console.error('Generation export failed:', error);
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
            <div className="space-y-3">
              <Label>{additionalDataLabel}</Label>
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="generation-include-advanced"
                  checked={includeAdvanced}
                  onCheckedChange={(checked) => setIncludeAdvanced(Boolean(checked))}
                />
                <Label htmlFor="generation-include-advanced" className="text-sm font-normal">
                  {includeAdvancedLabel}
                </Label>
              </div>
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

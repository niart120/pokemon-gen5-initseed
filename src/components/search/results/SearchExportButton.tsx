import { useState } from 'react';
import { Download, Copy, Check } from '@phosphor-icons/react';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import {
  formatSearchExportCopyLabel,
  formatSearchExportSummary,
  formatSearchExportTriggerLabel,
  searchExportAdditionalDataLabel,
  searchExportDialogTitle,
  searchExportDownloadLabel,
  searchExportFormatLabel,
  searchExportFormatOptions,
  searchExportFormatPlaceholder,
  searchExportIncludeDetailsLabel,
  searchExportIncludeHashLabel,
  searchExportIncludeMessageLabel,
} from '@/lib/i18n/strings/search-export';
import {
  exportSearchResults,
  downloadFile,
  copyToClipboard,
  generateFilename,
  type SearchExportOptions,
} from '@/lib/export/search-exporter';
import { MIME_TYPES } from '@/lib/export/file-utils';
import type { SearchResult } from '@/types/search';

interface SearchExportButtonProps {
  results: SearchResult[];
  disabled?: boolean;
}

export function SearchExportButton({ results, disabled = false }: SearchExportButtonProps) {
  const locale = useLocale();
  const [isOpen, setIsOpen] = useState(false);
  const [exportOptions, setExportOptions] = useState<SearchExportOptions>({
    format: 'csv',
    includeDetails: false,
    includeMessage: false,
    includeHash: true,
  });
  const [copied, setCopied] = useState(false);

  const handleExport = async (download: boolean = true) => {
    try {
      const content = exportSearchResults(results, exportOptions, locale);

      if (download) {
        const filename = generateFilename(exportOptions.format, 'pokemon-seeds');
        downloadFile(content, filename, MIME_TYPES[exportOptions.format]);
      } else {
        const success = await copyToClipboard(content);
        if (success) {
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
        }
      }

      setIsOpen(false);
    } catch (error) {
      console.error('Export failed:', error);
    }
  };

  const updateOption = <K extends keyof SearchExportOptions>(key: K, value: SearchExportOptions[K]) => {
    setExportOptions((prev) => ({ ...prev, [key]: value }));
  };

  const triggerLabel = formatSearchExportTriggerLabel(results.length, locale);
  const dialogTitle = resolveLocaleValue(searchExportDialogTitle, locale);
  const formatLabel = resolveLocaleValue(searchExportFormatLabel, locale);
  const formatPlaceholder = resolveLocaleValue(searchExportFormatPlaceholder, locale);
  const additionalDataLabel = resolveLocaleValue(searchExportAdditionalDataLabel, locale);
  const includeDetailsLabel = resolveLocaleValue(searchExportIncludeDetailsLabel, locale);
  const includeHashLabel = resolveLocaleValue(searchExportIncludeHashLabel, locale);
  const includeMessageLabel = resolveLocaleValue(searchExportIncludeMessageLabel, locale);
  const downloadLabel = resolveLocaleValue(searchExportDownloadLabel, locale);
  const copyLabel = formatSearchExportCopyLabel(copied, locale);
  const summaryLabel = formatSearchExportSummary(results.length, locale);

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
        <Button variant="outline" size="sm" disabled={disabled || results.length === 0} className="gap-2">
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
            <Label htmlFor="format">{formatLabel}</Label>
            <Select
              value={exportOptions.format}
              onValueChange={(value: 'csv' | 'json' | 'txt') => updateOption('format', value)}
            >
              <SelectTrigger id="format">
                <SelectValue placeholder={formatPlaceholder} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="csv">{resolveLocaleValue(searchExportFormatOptions.csv, locale)}</SelectItem>
                <SelectItem value="json">{resolveLocaleValue(searchExportFormatOptions.json, locale)}</SelectItem>
                <SelectItem value="txt">{resolveLocaleValue(searchExportFormatOptions.txt, locale)}</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-3">
            <Label>{additionalDataLabel}</Label>

            <div className="flex items-center space-x-2">
              <Checkbox
                id="details"
                checked={exportOptions.includeDetails}
                onCheckedChange={(checked) => updateOption('includeDetails', Boolean(checked))}
              />
              <Label htmlFor="details" className="text-sm font-normal">
                {includeDetailsLabel}
              </Label>
            </div>

            <div className="flex items-center space-x-2">
              <Checkbox
                id="hash"
                checked={exportOptions.includeHash}
                onCheckedChange={(checked) => updateOption('includeHash', Boolean(checked))}
              />
              <Label htmlFor="hash" className="text-sm font-normal">
                {includeHashLabel}
              </Label>
            </div>

            <div className="flex items-center space-x-2">
              <Checkbox
                id="message"
                checked={exportOptions.includeMessage}
                onCheckedChange={(checked) => updateOption('includeMessage', Boolean(checked))}
              />
              <Label htmlFor="message" className="text-sm font-normal">
                {includeMessageLabel}
              </Label>
            </div>
          </div>

          <div className="text-sm text-muted-foreground">{summaryLabel}</div>

          <div className="flex gap-2">
            <Button onClick={() => handleExport(true)} className="flex-1 gap-2">
              <Download size={16} />
              {downloadLabel}
            </Button>

            <Button variant="outline" onClick={() => handleExport(false)} className="flex-1 gap-2">
              {copied ? <Check size={16} /> : <Copy size={16} />}
              {copyLabel}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

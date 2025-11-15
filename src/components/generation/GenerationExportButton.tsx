import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Download, Copy, Check } from '@phosphor-icons/react';
import { exportGenerationResults } from '@/lib/export/generation-exporter';
import type { GenerationResult } from '@/types/generation';

type ExportFormat = 'csv' | 'json' | 'txt';

interface GenerationExportButtonProps {
  results: GenerationResult[];
  disabled?: boolean;
}

export function GenerationExportButton({ results, disabled = false }: GenerationExportButtonProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [format, setFormat] = useState<ExportFormat>('csv');
  const [copied, setCopied] = useState(false);

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
          Export ({results.length})
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Export Generation Results</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <Label htmlFor="generation-export-format">Export Format</Label>
            <Select value={format} onValueChange={value => setFormat(value as ExportFormat)}>
              <SelectTrigger id="generation-export-format">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="csv">CSV (Comma Separated Values)</SelectItem>
                <SelectItem value="json">JSON (JavaScript Object Notation)</SelectItem>
                <SelectItem value="txt">TXT (Plain Text)</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="text-sm text-muted-foreground">
            Exporting {results.length} result{results.length !== 1 ? 's' : ''}
          </div>
          <div className="flex gap-2">
            <Button onClick={() => handleExport(true)} className="flex-1 gap-2">
              <Download size={16} />
              Download File
            </Button>
            <Button
              variant="outline"
              onClick={() => handleExport(false)}
              className="flex-1 gap-2"
            >
              {copied ? <Check size={16} /> : <Copy size={16} />}
              {copied ? 'Copied!' : 'Copy to Clipboard'}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

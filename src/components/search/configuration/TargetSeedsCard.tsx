import React from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Trash, Upload, Download, Warning, Target, FileText } from '@phosphor-icons/react';
import { useAppStore } from '../../../store/app-store';
import { SeedCalculator } from '../../../lib/core/seed-calculator';
import { useResponsiveLayout } from '../../../hooks/use-mobile';
import { TemplateSelectionDialog } from './TemplateSelectionDialog';

export function TargetSeedsCard() {
  const {
    targetSeeds,
    setTargetSeeds,
    clearTargetSeeds,
    targetSeedInput,
    setTargetSeedInput,
  } = useAppStore();

  const { isStack } = useResponsiveLayout();
  const [parseErrors, setParseErrors] = React.useState<{ line: number; value: string; error: string }[]>([]);
  const [isTemplateDialogOpen, setIsTemplateDialogOpen] = React.useState(false);

  // Parse input and update seeds when input changes
  React.useEffect(() => {
    if (targetSeedInput.trim() === '') {
      setParseErrors([]);
      return;
    }

    const calculator = new SeedCalculator();
    const { validSeeds, errors } = calculator.parseTargetSeeds(targetSeedInput);
    setParseErrors(errors);
    
    if (errors.length === 0) {
      setTargetSeeds(validSeeds);
    }
  }, [targetSeedInput, setTargetSeeds]);

  const handleClearAll = () => {
    setTargetSeedInput('');
    clearTargetSeeds();
    setParseErrors([]);
  };

  const handleImportFromFile = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      setTargetSeedInput(content);
    };
    reader.readAsText(file);
    
    // Reset file input
    event.target.value = '';
  };

  const handleExportToFile = () => {
    if (targetSeeds.seeds.length === 0) return;

    const content = targetSeeds.seeds.map(seed => `0x${seed.toString(16).toUpperCase().padStart(8, '0')}`).join('\n');
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = 'target-seeds.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleApplyTemplate = (seeds: number[]) => {
    // Convert seeds to hex format and set as input
    const seedsText = seeds.map(seed => `0x${seed.toString(16).toUpperCase().padStart(8, '0')}`).join('\n');
    setTargetSeedInput(seedsText);
  };

  const exampleSeeds = [
    '0x12345678',
    'ABCDEF00', 
    '0xDEADBEEF',
  ];

  return (
    <>
      <PanelCard
      icon={<Target size={20} className="opacity-80" />}
      title="Target Seeds"
      headerActions={
        <div
          className="grid grid-cols-2 sm:grid-cols-4 gap-2 w-full"
          role="group"
          aria-label="Target Seeds operations"
        >
          <Button className="w-full" variant="outline" size="sm" onClick={() => setIsTemplateDialogOpen(true)}>
            <FileText size={14} className="mr-2" />
            Template
          </Button>
          <Button className="w-full" variant="outline" size="sm" onClick={() => document.getElementById('target-file-input')?.click()}>
            <Upload size={14} className="mr-2" />
            Import
          </Button>
          <Button
            className="w-full"
            variant="outline"
            size="sm"
            onClick={handleExportToFile}
            disabled={targetSeeds.seeds.length === 0}
          >
            <Download size={14} className="mr-2" />
            Export
          </Button>
          <Button
            className="w-full"
            variant="outline"
            size="sm"
            onClick={handleClearAll}
            disabled={targetSeeds.seeds.length === 0}
          >
            <Trash size={14} className="mr-2" />
            Clear
          </Button>
        </div>
      }
      className={isStack ? 'max-h-96' : 'min-h-64'}
      fullHeight={!isStack}
    >
        <p className="text-xs text-muted-foreground flex-shrink-0">
          Supports hex format with or without 0x prefix. One seed per line.
        </p>
        <Textarea
          id="seed-input"
          placeholder={`Enter seed values in hexadecimal format:\n${exampleSeeds.join('\n')}`}
          value={targetSeedInput}
          onChange={(e) => setTargetSeedInput(e.target.value)}
          className="flex-1 min-h-20 max-h-48 font-mono text-sm resize-none overflow-auto"
        />

        {/* Hidden file input */}
        <input
          id="target-file-input"
          type="file"
          accept=".txt,.csv"
          onChange={handleImportFromFile}
          className="hidden"
        />

        {/* Status */}
        <div className="flex items-center justify-between flex-shrink-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">Valid Seeds:</span>
            <Badge variant={targetSeeds.seeds.length > 0 ? "default" : "secondary"}>
              {targetSeeds.seeds.length}
            </Badge>
          </div>
          {parseErrors.length > 0 && (
            <Badge variant="destructive">
              {parseErrors.length} error{parseErrors.length !== 1 ? 's' : ''}
            </Badge>
          )}
        </div>

        {/* Parse Errors */}
        {parseErrors.length > 0 && (
          <Alert className="flex-shrink-0">
            <Warning size={14} />
            <AlertDescription>
              <div className="space-y-1">
                <p className="text-sm font-medium">Invalid seed format on the following lines:</p>
                <ul className="text-xs space-y-1">
                  {parseErrors.map((error, index) => (
                    <li key={index} className="font-mono">
                      Line {error.line}: "{error.value}" - {error.error}
                    </li>
                  ))}
                </ul>
              </div>
            </AlertDescription>
          </Alert>
        )}
      </PanelCard>

      {/* Template Selection Dialog */}
      <TemplateSelectionDialog
        isOpen={isTemplateDialogOpen}
        onOpenChange={setIsTemplateDialogOpen}
        onApplyTemplate={handleApplyTemplate}
      />
    </>
  );
}

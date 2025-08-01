import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Trash, Upload, Download, Warning, Target } from '@phosphor-icons/react';
import { useAppStore } from '../../../store/app-store';
import { SeedCalculator } from '../../../lib/core/seed-calculator';
import { useResponsiveLayout } from '../../../hooks/use-mobile';

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
  const calculator = new SeedCalculator();

  // Parse input and update seeds when input changes
  React.useEffect(() => {
    if (targetSeedInput.trim() === '') {
      setParseErrors([]);
      return;
    }

    const { validSeeds, errors } = calculator.parseTargetSeeds(targetSeedInput);
    setParseErrors(errors);
    
    if (errors.length === 0) {
      setTargetSeeds(validSeeds);
    }
  }, [targetSeedInput]);

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

  const exampleSeeds = [
    '0x12345678',
    'ABCDEF00', 
    '0xDEADBEEF',
  ];

  return (
    <Card className={`py-2 flex flex-col ${isStack ? 'max-h-96' : 'h-full min-h-64'}`}>
      <CardHeader className="pb-0 flex-shrink-0">
        <CardTitle className="flex items-center justify-between text-base">
          <div className="flex items-center gap-2">
            <Target size={20} />
            Target Seeds
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => document.getElementById('target-file-input')?.click()}>
              <Upload size={14} className="mr-2" />
              Import
            </Button>
            <Button 
              variant="outline" 
              size="sm" 
              onClick={handleExportToFile}
              disabled={targetSeeds.seeds.length === 0}
            >
              <Download size={14} className="mr-2" />
              Export
            </Button>
            <Button 
              variant="outline" 
              size="sm" 
              onClick={handleClearAll}
              disabled={targetSeeds.seeds.length === 0}
            >
              <Trash size={14} className="mr-2" />
              Clear
            </Button>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col min-h-0 space-y-2 overflow-y-auto">
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
      </CardContent>
    </Card>
  );
}

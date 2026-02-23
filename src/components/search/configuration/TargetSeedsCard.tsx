import React from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Trash, Warning, Target, FileText } from '@phosphor-icons/react';
import { useAppStore } from '../../../store/app-store';
import { SeedCalculator } from '../../../lib/core/seed-calculator';
import { useResponsiveLayout } from '../../../hooks/use-mobile';
import { TemplateSelectionDialog } from './TemplateSelectionDialog';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import {
  formatTargetSeedsErrorBadge,
  formatTargetSeedsErrorLine,
  formatTargetSeedsPlaceholder,
  targetSeedsAriaLabel,
  targetSeedsClearButtonLabel,
  targetSeedsPanelTitle,
  targetSeedsParseErrorSummary,
  targetSeedsSupportsHexHint,
  targetSeedsTemplateButtonLabel,
  targetSeedsValidSeedsLabel,
} from '@/lib/i18n/strings/search-target-seeds';

export function TargetSeedsCard() {
  const {
    targetSeeds,
    setTargetSeeds,
    clearTargetSeeds,
    targetSeedInput,
    setTargetSeedInput,
  } = useAppStore();

  const { isStack } = useResponsiveLayout();
  const locale = useLocale();
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
  const placeholderText = formatTargetSeedsPlaceholder(exampleSeeds, locale);
  const operationsLabel = resolveLocaleValue(targetSeedsAriaLabel, locale);
  const colon = locale === 'ja' ? '：' : ':';

  return (
    <>
      <PanelCard
        icon={<Target size={20} className="opacity-80" />}
        title={resolveLocaleValue(targetSeedsPanelTitle, locale)}
        headerActions={
        <div
          className="flex gap-2"
          role="group"
          aria-label={operationsLabel}
        >
          <Button className="flex-1" variant="outline" size="sm" onClick={() => setIsTemplateDialogOpen(true)}>
            <FileText size={14} className="mr-2" />
            {resolveLocaleValue(targetSeedsTemplateButtonLabel, locale)}
          </Button>
          <Button
            className="flex-1"
            variant="outline"
            size="sm"
            onClick={handleClearAll}
            disabled={targetSeeds.seeds.length === 0}
          >
            <Trash size={14} className="mr-2" />
            {resolveLocaleValue(targetSeedsClearButtonLabel, locale)}
          </Button>
        </div>
      }
        className={isStack ? 'max-h-96' : 'min-h-64'}
        fullHeight={!isStack}
      >
        <p className="text-xs text-muted-foreground flex-shrink-0">
          {resolveLocaleValue(targetSeedsSupportsHexHint, locale)}
        </p>
        <Textarea
          id="seed-input"
          placeholder={placeholderText}
          value={targetSeedInput}
          onChange={(e) => setTargetSeedInput(e.target.value)}
          className="flex-1 min-h-20 max-h-48 font-mono text-sm resize-none overflow-auto"
        />

        {/* Status */}
        <div className="flex items-center justify-between flex-shrink-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">
              {resolveLocaleValue(targetSeedsValidSeedsLabel, locale)}{colon}
            </span>
            <Badge variant={targetSeeds.seeds.length > 0 ? "default" : "secondary"}>
              {targetSeeds.seeds.length}
            </Badge>
          </div>
          {parseErrors.length > 0 && (
            <Badge variant="destructive">
              {formatTargetSeedsErrorBadge(parseErrors.length, locale)}
            </Badge>
          )}
        </div>

        {/* Parse Errors */}
        {parseErrors.length > 0 && (
          <Alert className="flex-shrink-0">
            <Warning size={14} />
            <AlertDescription>
              <div className="space-y-1">
                <p className="text-sm font-medium">
                  {resolveLocaleValue(targetSeedsParseErrorSummary, locale)}
                </p>
                <ul className="text-xs space-y-1">
                  {parseErrors.map((error, index) => (
                    <li key={index} className="font-mono">
                      {formatTargetSeedsErrorLine(error.line, error.value, error.error, locale)}
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

import { useMemo } from 'react';
import { Eye, Copy } from 'lucide-react';
import { Button } from '../../ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '../../ui/dialog';
import { Label } from '../../ui/label';
import { Tooltip, TooltipContent, TooltipTrigger } from '../../ui/tooltip';
import { toast } from 'sonner';
import { lcgSeedToHex, lcgSeedToMtSeed } from '@/lib/utils/lcg-seed';
import { formatKeyInputForDisplay } from '@/lib/utils/key-input';
import { getIvTooltipEntries } from '@/lib/utils/individual-values-display';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import {
  clipboardUnavailable,
  copyMtSeedHint,
  bootTimingCopyHint,
  copyToGenerationPanelHint,
  dateTimeLabel,
  detailsButtonLabel,
  generatedMessageLabel,
  hardwareLabel,
  keyInputLabel,
  lcgSeedLabel,
  mtSeedCopyFailure,
  mtSeedCopySuccess,
  mtSeedLabel,
  resultDetailsTitle,
  romLabel,
  sha1HashLabel,
  timer0Label,
  vcountLabel,
} from '@/lib/i18n/strings/search-results';
import {
  formatBootTimestampDisplay,
  formatTimer0Hex,
  formatVCountHex,
} from '@/lib/generation/result-formatters';
import type { InitialSeedResult } from '../../../types/search';
import { useResultDetailsClipboard } from '@/hooks/search/useResultDetailsClipboard';

interface ResultDetailsDialogProps {
  result: InitialSeedResult | null;
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ResultDetailsDialog({
  result,
  isOpen,
  onOpenChange,
}: ResultDetailsDialogProps) {
  const locale = useLocale();
  const { copySeedToGeneration, copyBootTimingToGeneration } = useResultDetailsClipboard(locale);

  const handleCopyLcgSeed = () => {
    if (!result) return;
    copySeedToGeneration(result);
  };

  const lcgSeedHex = result ? lcgSeedToHex(result.lcgSeed) : '';
  const mtSeedHex = result ? `0x${result.seed.toString(16).toUpperCase().padStart(8, '0')}` : '';
  const lcgTooltipEntries = useMemo(() => {
    if (!result) return [];
    return getIvTooltipEntries(lcgSeedToMtSeed(result.lcgSeed), locale);
  }, [result, locale]);
  const mtTooltipEntries = useMemo(() => {
    if (!result) return [];
    return getIvTooltipEntries(result.seed >>> 0, locale);
  }, [result, locale]);

  const handleCopyMtSeed = async () => {
    if (!result) return;
    if (typeof navigator === 'undefined' || !navigator.clipboard?.writeText) {
      toast.error(resolveLocaleValue(clipboardUnavailable, locale));
      return;
    }
    try {
      await navigator.clipboard.writeText(mtSeedHex);
      toast.success(resolveLocaleValue(mtSeedCopySuccess, locale));
    } catch {
      toast.error(resolveLocaleValue(mtSeedCopyFailure, locale));
    }
  };

  const handleCopyBootTiming = () => {
    if (!result) return;
    copyBootTimingToGeneration(result);
  };

  if (!result) return null;

  const keyInputDisplay = formatKeyInputForDisplay(result.keyCode, result.keyInputNames);
  const timer0Display = formatTimer0Hex(result?.timer0);
  const vcountDisplay = formatVCountHex(result?.vcount);
  const formattedDateTime = formatBootTimestampDisplay(result.datetime, locale);
  const bootTimingHint = resolveLocaleValue(bootTimingCopyHint, locale);

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
  <DialogContent className="sm:max-w-xl md:max-w-2xl lg:max-w-3xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{resolveLocaleValue(resultDetailsTitle, locale)}</DialogTitle>
        </DialogHeader>
        <div className="space-y-6">
          {/* Basic Info */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label>{resolveLocaleValue(lcgSeedLabel, locale)}</Label>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div 
                    className="font-mono text-lg cursor-pointer hover:bg-accent p-2 rounded flex items-center gap-2 group"
                    onClick={handleCopyLcgSeed}
                    title={resolveLocaleValue(copyToGenerationPanelHint, locale)}
                  >
                    {lcgSeedHex}
                    <Copy size={16} className="opacity-0 group-hover:opacity-100 transition-opacity" />
                  </div>
                </TooltipTrigger>
                <TooltipContent side="top" className="space-y-1 text-left max-w-[90vw]">
                  {lcgTooltipEntries.map(entry => (
                    <div key={entry.label} className="space-y-0.5">
                      <div className="font-semibold leading-tight">{entry.label}</div>
                      <div className="font-mono leading-tight">{entry.spread}</div>
                      <div className="font-mono text-[10px] text-muted-foreground leading-tight">{entry.pattern}</div>
                    </div>
                  ))}
                </TooltipContent>
              </Tooltip>
              <div className="text-xs text-muted-foreground">{resolveLocaleValue(copyToGenerationPanelHint, locale)}</div>
            </div>
            <div>
              <Label>{resolveLocaleValue(mtSeedLabel, locale)}</Label>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div
                    className="font-mono text-lg cursor-pointer hover:bg-accent p-2 rounded flex items-center gap-2 group"
                    onClick={handleCopyMtSeed}
                    title={resolveLocaleValue(copyMtSeedHint, locale)}
                  >
                    {mtSeedHex}
                    <Copy size={16} className="opacity-0 group-hover:opacity-100 transition-opacity" />
                  </div>
                </TooltipTrigger>
                <TooltipContent side="top" className="space-y-1 text-left max-w-[90vw]">
                  {mtTooltipEntries.map(entry => (
                    <div key={entry.label} className="space-y-0.5">
                      <div className="font-semibold leading-tight">{entry.label}</div>
                      <div className="font-mono leading-tight">{entry.spread}</div>
                      <div className="font-mono text-[10px] text-muted-foreground leading-tight">{entry.pattern}</div>
                    </div>
                  ))}
                </TooltipContent>
              </Tooltip>
              <div className="text-xs text-muted-foreground">{resolveLocaleValue(copyMtSeedHint, locale)}</div>
            </div>
          </div>

          {/* Date/Time */}
          <div>
            <Label>{resolveLocaleValue(dateTimeLabel, locale)}</Label>
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  className="font-mono w-full text-left cursor-pointer hover:bg-accent p-2 rounded"
                  onClick={handleCopyBootTiming}
                  title={bootTimingHint}
                >
                  {formattedDateTime}
                </button>
              </TooltipTrigger>
              <TooltipContent side="top">
                {bootTimingHint}
              </TooltipContent>
            </Tooltip>
            <div className="text-xs text-muted-foreground">{bootTimingHint}</div>
          </div>

          {/* Parameters */}
          <div className="grid grid-cols-4 gap-4">
            <div>
              <Label>{resolveLocaleValue(timer0Label, locale)}</Label>
              <div className="font-mono">{timer0Display}</div>
            </div>
            <div>
              <Label>{resolveLocaleValue(vcountLabel, locale)}</Label>
              <div className="font-mono">{vcountDisplay}</div>
            </div>
            <div>
              <Label>{resolveLocaleValue(romLabel, locale)}</Label>
              <div>{result.conditions.romVersion} {result.conditions.romRegion}</div>
            </div>
            <div>
              <Label>{resolveLocaleValue(hardwareLabel, locale)}</Label>
              <div>{result.conditions.hardware}</div>
            </div>
          </div>

          {/* Key Input */}
          <div>
            <Label>{resolveLocaleValue(keyInputLabel, locale)}</Label>
            <div className="font-mono text-sm font-arrows">
              {keyInputDisplay}
            </div>
          </div>

          {/* SHA-1 Hash */}
          <div>
            <Label>{resolveLocaleValue(sha1HashLabel, locale)}</Label>
            <div className="font-mono text-sm break-all p-2 bg-muted rounded">
              {result.sha1Hash}
            </div>
          </div>

          {/* Message Array */}
          <div>
            <Label>{resolveLocaleValue(generatedMessageLabel, locale)}</Label>
            <div className="grid grid-cols-4 gap-2 mt-2">
              {result.message.map((word, index) => (
                <div key={index} className="text-center">
                  <div className="text-xs text-muted-foreground">data[{index}]</div>
                  <div className="font-mono text-sm p-1 bg-muted rounded">
                    0x{word.toString(16).toUpperCase().padStart(8, '0')}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

// Trigger button component for use in the table
interface ResultDetailsButtonProps {
  result: InitialSeedResult;
  onClick: () => void;
}

export function ResultDetailsButton({ result: _result, onClick }: ResultDetailsButtonProps) {
  const locale = useLocale();
  return (
    <Button 
      variant="outline" 
      size="sm"
      onClick={onClick}
    >
      <Eye size={16} className="mr-1" />
      {resolveLocaleValue(detailsButtonLabel, locale)}
    </Button>
  );
}

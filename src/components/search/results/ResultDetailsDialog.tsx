import { useMemo } from 'react';
import { Eye, Copy } from 'lucide-react';
import { Button } from '../../ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '../../ui/dialog';
import { Label } from '../../ui/label';
import { Tooltip, TooltipContent, TooltipTrigger } from '../../ui/tooltip';
import { toast } from 'sonner';
import { lcgSeedToHex, lcgSeedToMtSeed } from '@/lib/utils/lcg-seed';
import { keyCodeToNames } from '@/lib/utils/key-input';
import { useAppStore } from '@/store/app-store';
import { getIvTooltipEntries } from '@/lib/utils/individual-values-display';
import { useLocale } from '@/lib/i18n/locale-context';
import type { InitialSeedResult } from '../../../types/search';

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
  const { setDraftParams } = useAppStore();
  const locale = useLocale();

  const formatDateTime = (date: Date): string => {
    return `${date.getFullYear()}/${String(date.getMonth() + 1).padStart(2, '0')}/${String(date.getDate()).padStart(2, '0')} ${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}:${String(date.getSeconds()).padStart(2, '0')}`;
  };

  const handleCopyLcgSeed = () => {
    if (!result) return;
    
    const lcgSeedHex = lcgSeedToHex(result.lcgSeed);
    
    // Copy to Generation Panel
    setDraftParams({
      baseSeedHex: lcgSeedHex,
    });
    
    toast.success('LCG Seed copied to Generation Panel');
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
      toast.error('Clipboard is not available');
      return;
    }
    try {
      await navigator.clipboard.writeText(mtSeedHex);
      toast.success('MT Seed copied to clipboard');
    } catch {
      toast.error('Failed to copy MT Seed');
    }
  };

  if (!result) return null;

  const keyNames = result.keyCode != null ? keyCodeToNames(result.keyCode) : [];
  const keyInputDisplay = result.keyCode == null
    ? 'Unavailable'
    : keyNames.length > 0
      ? keyNames.join(', ')
      : 'No keys';

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
  <DialogContent className="sm:max-w-xl md:max-w-2xl lg:max-w-3xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Seed Result Details</DialogTitle>
        </DialogHeader>
        <div className="space-y-6">
          {/* Basic Info */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label>LCG Seed</Label>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div 
                    className="font-mono text-lg cursor-pointer hover:bg-accent p-2 rounded flex items-center gap-2 group"
                    onClick={handleCopyLcgSeed}
                    title="Click to copy to Generation Panel"
                  >
                    {lcgSeedHex}
                    <Copy size={16} className="opacity-0 group-hover:opacity-100 transition-opacity" />
                  </div>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="space-y-1 text-left">
                  {lcgTooltipEntries.map(entry => (
                    <div key={entry.label} className="space-y-0.5">
                      <div className="font-semibold leading-tight">{entry.label}</div>
                      <div className="font-mono leading-tight">{entry.spread}</div>
                      <div className="font-mono text-[10px] text-muted-foreground leading-tight">{entry.pattern}</div>
                    </div>
                  ))}
                </TooltipContent>
              </Tooltip>
              <div className="text-xs text-muted-foreground">Click to copy to Generation Panel</div>
            </div>
            <div>
              <Label>MT Seed</Label>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div
                    className="font-mono text-lg cursor-pointer hover:bg-accent p-2 rounded flex items-center gap-2 group"
                    onClick={handleCopyMtSeed}
                    title="Click to copy MT Seed"
                  >
                    {mtSeedHex}
                    <Copy size={16} className="opacity-0 group-hover:opacity-100 transition-opacity" />
                  </div>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="space-y-1 text-left">
                  {mtTooltipEntries.map(entry => (
                    <div key={entry.label} className="space-y-0.5">
                      <div className="font-semibold leading-tight">{entry.label}</div>
                      <div className="font-mono leading-tight">{entry.spread}</div>
                      <div className="font-mono text-[10px] text-muted-foreground leading-tight">{entry.pattern}</div>
                    </div>
                  ))}
                </TooltipContent>
              </Tooltip>
              <div className="text-xs text-muted-foreground">Click to copy MT Seed</div>
            </div>
          </div>

          {/* Date/Time */}
          <div>
            <Label>Date/Time</Label>
            <div className="font-mono">
              {formatDateTime(result.datetime)}
            </div>
          </div>

          {/* Parameters */}
          <div className="grid grid-cols-4 gap-4">
            <div>
              <Label>Timer0</Label>
              <div className="font-mono">0x{result.timer0.toString(16).toUpperCase().padStart(4, '0')}</div>
            </div>
            <div>
              <Label>VCount</Label>
              <div className="font-mono">0x{result.vcount.toString(16).toUpperCase().padStart(2, '0')}</div>
            </div>
            <div>
              <Label>ROM</Label>
              <div>{result.conditions.romVersion} {result.conditions.romRegion}</div>
            </div>
            <div>
              <Label>Hardware</Label>
              <div>{result.conditions.hardware}</div>
            </div>
          </div>

          {/* Key Input */}
          <div>
            <Label>Key Input</Label>
            <div className="font-mono text-sm font-arrows">
              {keyInputDisplay}
            </div>
          </div>

          {/* SHA-1 Hash */}
          <div>
            <Label>SHA-1 Hash</Label>
            <div className="font-mono text-sm break-all p-2 bg-muted rounded">
              {result.sha1Hash}
            </div>
          </div>

          {/* Message Array */}
          <div>
            <Label>Generated Message (32-bit words)</Label>
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
  return (
    <Button 
      variant="outline" 
      size="sm"
      onClick={onClick}
    >
      <Eye size={16} className="mr-1" />
      Details
    </Button>
  );
}

/**
 * EggResultDetailsDialog
 * Search(Egg)パネルの結果詳細ダイアログ
 * SearchPanel の ResultDetailsDialog パターンに準拠
 */

import { useMemo } from 'react';
import { Copy } from 'lucide-react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { toast } from 'sonner';
import { formatKeyInputForDisplay } from '@/lib/utils/key-input';
import { getIvTooltipEntries } from '@/lib/utils/individual-values-display';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import { natureName } from '@/lib/utils/format-display';
import { hiddenPowerTypeNames } from '@/lib/i18n/strings/hidden-power';
import {
  clipboardUnavailable,
  copyMtSeedHint,
  bootTimingCopyHint,
  copyToGenerationPanelHint,
  dateTimeLabel,
  hardwareLabel,
  keyInputLabel,
  lcgSeedLabel,
  mtSeedCopyFailure,
  mtSeedCopySuccess,
  mtSeedLabel,
  romLabel,
  sha1HashLabel,
  generatedMessageLabel,
  timer0Label,
  vcountLabel,
} from '@/lib/i18n/strings/search-results';
import {
  formatBootTimestampDisplay,
  formatTimer0Hex,
  formatVCountHex,
} from '@/lib/generation/result-formatters';
import { eggSearchResultsTableHeaders, eggSearchAbilityOptions, eggSearchGenderOptions } from '@/lib/i18n/strings/egg-search';
import { eggResultDetailsTitle, pidLabel, advanceLabel, stableLabel } from '@/lib/i18n/strings/egg-result-details';
import { useEggResultDetailsClipboard } from '@/hooks/egg-search/useEggResultDetailsClipboard';
import { useEggBootTimingSearchStore } from '@/store/egg-boot-timing-search-store';
import { SeedCalculator } from '@/lib/core/seed-calculator';
import type { EggBootTimingSearchResult } from '@/types/egg-boot-timing-search';
import type { SearchConditions } from '@/types/search';
import { IV_UNKNOWN } from '@/types/egg';

interface EggResultDetailsDialogProps {
  result: EggBootTimingSearchResult | null;
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
}

// SHA-1 Hash / Message 生成用のシングルトン計算器
const seedCalculator = new SeedCalculator();

export function EggResultDetailsDialog({
  result,
  isOpen,
  onOpenChange,
}: EggResultDetailsDialogProps) {
  const locale = useLocale();
  const { copySeedToGeneration, copyBootTimingToGeneration } = useEggResultDetailsClipboard(locale);
  const draftParams = useEggBootTimingSearchStore((s) => s.draftParams);

  const handleCopyLcgSeed = () => {
    if (!result) return;
    copySeedToGeneration(result);
  };

  const lcgSeedHex = result?.lcgSeedHex ?? '';
  const mtSeedHex = result?.egg.egg.mtSeedHex ?? '-';

  // MT Seed を数値に変換（IV Tooltip 用）
  const mtSeedNumber = useMemo(() => {
    if (!result?.egg.egg.mtSeedHex) return 0;
    return parseInt(result.egg.egg.mtSeedHex.replace(/^0x/i, ''), 16) >>> 0;
  }, [result]);

  // LCG Seed を数値に変換（IV Tooltip 用）
  const lcgSeedNumber = useMemo(() => {
    if (!result?.lcgSeedHex) return 0n;
    return BigInt(result.lcgSeedHex.startsWith('0x') ? result.lcgSeedHex : `0x${result.lcgSeedHex}`);
  }, [result]);

  // LCG Seed → MT Seed 変換（IV Tooltip 用）
  const lcgDerivedMtSeed = useMemo(() => {
    if (!lcgSeedNumber) return 0;
    const multiplier = 0x5D588B656C078965n;
    const addValue = 0x269EC3n;
    const seed = lcgSeedNumber * multiplier + addValue;
    return Number((seed >> 32n) & 0xFFFFFFFFn);
  }, [lcgSeedNumber]);

  const lcgTooltipEntries = useMemo(() => {
    if (!result) return [];
    return getIvTooltipEntries(lcgDerivedMtSeed, locale);
  }, [result, lcgDerivedMtSeed, locale]);

  const mtTooltipEntries = useMemo(() => {
    if (!result) return [];
    return getIvTooltipEntries(mtSeedNumber, locale);
  }, [result, mtSeedNumber, locale]);

  // SHA-1 Hash / Message を都度生成
  const { sha1Hash, message } = useMemo(() => {
    if (!result) return { sha1Hash: '', message: [] };
    
    // SearchConditions を構築
    const conditions: SearchConditions = {
      romVersion: draftParams.romVersion,
      romRegion: draftParams.romRegion,
      hardware: draftParams.hardware,
      timer0VCountConfig: {
        useAutoConfiguration: false,
        timer0Range: draftParams.timer0Range,
        vcountRange: draftParams.vcountRange,
      },
      timeRange: draftParams.timeRange,
      dateRange: {
        startYear: draftParams.dateRange.startYear,
        endYear: draftParams.dateRange.endYear,
        startMonth: draftParams.dateRange.startMonth,
        endMonth: draftParams.dateRange.endMonth,
        startDay: draftParams.dateRange.startDay,
        endDay: draftParams.dateRange.endDay,
        startHour: draftParams.timeRange.hour.start,
        endHour: draftParams.timeRange.hour.end,
        startMinute: draftParams.timeRange.minute.start,
        endMinute: draftParams.timeRange.minute.end,
        startSecond: draftParams.timeRange.second.start,
        endSecond: draftParams.timeRange.second.end,
      },
      keyInput: draftParams.keyInputMask,
      macAddress: [...draftParams.macAddress],
    };

    try {
      const msg = seedCalculator.generateMessage(
        conditions,
        result.boot.timer0,
        result.boot.vcount,
        result.boot.datetime,
        result.boot.keyCode
      );
      const { hash } = seedCalculator.calculateSeed(msg);
      return { sha1Hash: hash, message: msg };
    } catch {
      return { sha1Hash: '', message: [] };
    }
  }, [result, draftParams]);

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

  const keyInputDisplay = formatKeyInputForDisplay(result.boot.keyCode, result.boot.keyInputNames);
  const timer0Display = formatTimer0Hex(result.boot.timer0);
  const vcountDisplay = formatVCountHex(result.boot.vcount);
  const formattedDateTime = formatBootTimestampDisplay(result.boot.datetime, locale);
  const bootTimingHint = resolveLocaleValue(bootTimingCopyHint, locale);

  // Egg specific displays
  const abilityOptions = resolveLocaleValue(eggSearchAbilityOptions, locale);
  const genderOptions = resolveLocaleValue(eggSearchGenderOptions, locale);
  const hpTypeNames = hiddenPowerTypeNames[locale] ?? hiddenPowerTypeNames.en;

  const formatIv = (iv: number): string => {
    return iv === IV_UNKNOWN ? '?' : String(iv);
  };

  const formatIVs = (ivs: readonly [number, number, number, number, number, number]): string => {
    return ivs.map(formatIv).join('-');
  };

  const getShinyDisplay = (shiny: number): { text: string; className: string } => {
    switch (shiny) {
      case 2:
        return { text: '★', className: 'text-yellow-500' };
      case 1:
        return { text: '◇', className: 'text-blue-500' };
      default:
        return { text: '-', className: 'text-muted-foreground' };
    }
  };

  const getAbilityDisplay = (ability: 0 | 1 | 2): string => {
    return abilityOptions[String(ability) as '0' | '1' | '2'];
  };

  const getGenderDisplay = (gender: 'male' | 'female' | 'genderless'): string => {
    return genderOptions[gender];
  };

  const getHiddenPowerDisplay = (): string => {
    const hpInfo = result.egg.egg.hiddenPower;
    if (hpInfo.type === 'unknown') return '-';
    return `${hpTypeNames[hpInfo.hpType]} ${hpInfo.power}`;
  };

  const shinyDisplay = getShinyDisplay(result.egg.egg.shiny);

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-xl md:max-w-2xl lg:max-w-3xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{resolveLocaleValue(eggResultDetailsTitle, locale)}</DialogTitle>
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
              <div>{draftParams.romVersion} {draftParams.romRegion}</div>
            </div>
            <div>
              <Label>{resolveLocaleValue(hardwareLabel, locale)}</Label>
              <div>{draftParams.hardware}</div>
            </div>
          </div>

          {/* Egg Info */}
          <div className="grid grid-cols-4 gap-4">
            <div>
              <Label>{resolveLocaleValue(advanceLabel, locale)}</Label>
              <div className="font-mono">{result.egg.advance}</div>
            </div>
            <div>
              <Label>{resolveLocaleValue(stableLabel, locale)}</Label>
              <div>{result.isStable ? '○' : '×'}</div>
            </div>
            <div>
              <Label>{eggSearchResultsTableHeaders.nature[locale]}</Label>
              <div>{natureName(result.egg.egg.nature, locale)}</div>
            </div>
            <div>
              <Label>{eggSearchResultsTableHeaders.shiny[locale]}</Label>
              <div className={shinyDisplay.className}>{shinyDisplay.text}</div>
            </div>
          </div>

          <div className="grid grid-cols-4 gap-4">
            <div>
              <Label>{eggSearchResultsTableHeaders.ability[locale]}</Label>
              <div>{getAbilityDisplay(result.egg.egg.ability)}</div>
            </div>
            <div>
              <Label>{eggSearchResultsTableHeaders.gender[locale]}</Label>
              <div>{getGenderDisplay(result.egg.egg.gender)}</div>
            </div>
            <div>
              <Label>{eggSearchResultsTableHeaders.ivs[locale]}</Label>
              <div className="font-mono">{formatIVs(result.egg.egg.ivs)}</div>
            </div>
            <div>
              <Label>{eggSearchResultsTableHeaders.hiddenPower[locale]}</Label>
              <div>{getHiddenPowerDisplay()}</div>
            </div>
          </div>

          {/* PID */}
          <div>
            <Label>{resolveLocaleValue(pidLabel, locale)}</Label>
            <div className="font-mono text-sm p-2 bg-muted rounded">
              0x{result.egg.egg.pid.toString(16).toUpperCase().padStart(8, '0')}
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
          {sha1Hash && (
            <div>
              <Label>{resolveLocaleValue(sha1HashLabel, locale)}</Label>
              <div className="font-mono text-sm break-all p-2 bg-muted rounded">
                {sha1Hash}
              </div>
            </div>
          )}

          {/* Message Array */}
          {message.length > 0 && (
            <div>
              <Label>{resolveLocaleValue(generatedMessageLabel, locale)}</Label>
              <div className="grid grid-cols-4 gap-2 mt-2">
                {message.map((word, index) => (
                  <div key={index} className="text-center">
                    <div className="text-xs text-muted-foreground">data[{index}]</div>
                    <div className="font-mono text-sm p-1 bg-muted rounded">
                      0x{word.toString(16).toUpperCase().padStart(8, '0')}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

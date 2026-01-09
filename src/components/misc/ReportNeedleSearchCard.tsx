/**
 * ReportNeedleSearchCard.tsx
 * レポ針検索カード（Boot-Timing / LCG Seed）
 */

import React, { useEffect, useMemo, useState, useCallback } from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Separator } from '@/components/ui/separator';
import { TimeInputHms } from '@/components/ui/time-input-hms';
import { DATE_INPUT_MAX, DATE_INPUT_MIN } from '@/components/ui/date-input-constraints';
import { KeyInputDialog } from '@/components/keys';
import {
  Play,
  Pause,
  Square,
  ArrowCounterClockwise,
  Copy,
  Compass,
  GameController,
} from '@phosphor-icons/react';
import { useLocale } from '@/lib/i18n/locale-context';
import {
  reportNeedleCardTitle,
  reportNeedleStatusPrefix,
  reportNeedleButtonLabels,
  reportNeedleParamLabels,
  reportNeedleResultsLabel,
  reportNeedleResultsEmpty,
  reportNeedleResultsSearching,
  reportNeedleTableHeaders,
  getReportNeedleStatusLabel,
  getReportNeedleModeLabel,
  reportNeedleKeyLabels,
  reportNeedleStartupPlaceholders,
} from '@/lib/i18n/strings/report-needle-search';
import { formatTimer0Hex, formatVCountHex } from '@/lib/generation/result-formatters';
import { useReportNeedleSearchStore } from '@/store/report-needle-search-store';
import { useAppStore } from '@/store/app-store';
import { formatKeyInputForDisplay, keyMaskToNames, toggleKeyInMask, KEY_INPUT_DEFAULT, type KeyName } from '@/lib/utils/key-input';
import type { ReportNeedleSearchMode } from '@/types/report-needle-search';

function toLocalDateTimeParts(iso?: string): { dateValue: string; timeValue: string } {
  if (!iso) return { dateValue: '', timeValue: '' };
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return { dateValue: '', timeValue: '' };
  const pad = (value: number) => value.toString().padStart(2, '0');
  const year = date.getFullYear();
  const month = pad(date.getMonth() + 1);
  const day = pad(date.getDate());
  const hours = pad(date.getHours());
  const minutes = pad(date.getMinutes());
  const seconds = pad(date.getSeconds());
  return {
    dateValue: `${year}-${month}-${day}`,
    timeValue: `${hours}:${minutes}:${seconds}`,
  };
}

export const ReportNeedleSearchCard: React.FC = () => {
  const locale = useLocale();
  const {
    draft,
    status,
    results,
    validationErrors,
    errorMessage,
    setMode,
    updateNeedleValue,
    appendNeedleDigit,
    updateRange,
    updateAdvanceRange,
    setStartupDate,
    setStartupTime,
    setStartupKeyMask,
    setInitialSeedHex,
    applyProfileRanges,
    startSearch,
    pauseSearch,
    resumeSearch,
    stopSearch,
    reset,
  } = useReportNeedleSearchStore();

  const profiles = useAppStore((s) => s.profiles);
  const activeProfileId = useAppStore((s) => s.activeProfileId);

  useEffect(() => {
    const profile = profiles.find((p) => p.id === activeProfileId) ?? profiles[0];
    if (profile) {
      applyProfileRanges(profile);
    }
  }, [profiles, activeProfileId, applyProfileRanges]);

  const isSearchActive = status === 'running' || status === 'paused' || status === 'starting' || status === 'stopping';
  const isIdle = status === 'idle' || status === 'completed' || status === 'error';
  const canPause = status === 'running';
  const canResume = status === 'paused';

  const statusDisplay = getReportNeedleStatusLabel(status, locale);
  const { dateValue: startupDateValue, timeValue: startupTimeValue } = useMemo(
    () => toLocalDateTimeParts(draft.startup.timestampIso),
    [draft.startup.timestampIso],
  );

  const [isKeyDialogOpen, setKeyDialogOpen] = useState(false);
  const [tempKeyMask, setTempKeyMask] = useState(draft.startup.keyMask);

  useEffect(() => {
    if (!isKeyDialogOpen) {
      setTempKeyMask(draft.startup.keyMask);
    }
  }, [draft.startup.keyMask, isKeyDialogOpen]);

  useEffect(() => {
    if (isSearchActive && isKeyDialogOpen) {
      setKeyDialogOpen(false);
    }
  }, [isSearchActive, isKeyDialogOpen]);

  const keyDisplay = useMemo(
    () => formatKeyInputForDisplay(null, keyMaskToNames(draft.startup.keyMask), reportNeedleKeyLabels.displayPlaceholder[locale]),
    [draft.startup.keyMask, locale],
  );
  const availableKeys = useMemo(() => keyMaskToNames(tempKeyMask), [tempKeyMask]);

  const handleOpenKeyDialog = useCallback(() => {
    if (isSearchActive) return;
    setTempKeyMask(draft.startup.keyMask);
    setKeyDialogOpen(true);
  }, [draft.startup.keyMask, isSearchActive]);

  const handleKeyDialogOpenChange = useCallback((open: boolean) => {
    if (isSearchActive && open) return;
    setKeyDialogOpen(open);
    if (!open) {
      setTempKeyMask(draft.startup.keyMask);
    }
  }, [draft.startup.keyMask, isSearchActive]);

  const handleToggleKey = useCallback((key: KeyName) => {
    setTempKeyMask(prev => toggleKeyInMask(prev, key));
  }, []);

  const handleResetKeys = useCallback(() => {
    setTempKeyMask(KEY_INPUT_DEFAULT);
  }, []);

  const handleApplyKeys = useCallback(() => {
    setStartupKeyMask(tempKeyMask);
    setKeyDialogOpen(false);
  }, [setStartupKeyMask, tempKeyMask]);

  const handleStartupDateInput = useCallback((value: string) => {
    setStartupDate(value);
  }, [setStartupDate]);

  const handleStartupTimeInput = useCallback((value: string) => {
    setStartupTime(value);
  }, [setStartupTime]);

  const combinedNeedle = useMemo(() => draft.needleValue, [draft.needleValue]);

  const handleStart = async () => {
    await startSearch();
  };

  const handleCopyResults = () => {
    if (!results.length) return;
    const lines = results.map((r) =>
      `${r.consumptionIndex}\t${formatTimer0Hex(r.timer0)}\t${formatVCountHex(r.vcount)}\t${r.initialSeed}\t${r.currentSeed}`
    );
    navigator.clipboard.writeText(lines.join('\n'));
  };

  const handleModeChange = (value: string) => {
    if (value === 'startup' || value === 'initial-seed') {
      setMode(value as ReportNeedleSearchMode);
    }
  };

  const renderResultsTable = () => {
    if (results.length === 0) {
      return (
        <div className="flex h-24 items-center justify-center text-muted-foreground text-xs">
          {status === 'running' || status === 'starting'
            ? reportNeedleResultsSearching[locale]
            : reportNeedleResultsEmpty[locale]}
        </div>
      );
    }

    const headers = reportNeedleTableHeaders[locale];

    return (
      <div className="overflow-auto max-h-64">
        <Table className="min-w-[520px] text-xs">
          <TableHeader className="sticky top-0 bg-muted text-xs">
            <TableRow className="text-left border-0">
              <TableHead scope="col" className="px-2 py-1 font-medium select-none">
                {headers.position}
              </TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium select-none">
                {headers.timer0}
              </TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium select-none">
                {headers.vcount}
              </TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium select-none">
                {headers.initialSeed}
              </TableHead>
              <TableHead scope="col" className="px-2 py-1 font-medium select-none">
                {headers.currentSeed}
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {results.map((result, idx) => (
              <TableRow key={`${result.consumptionIndex}-${result.timer0}-${result.vcount}-${idx}`} className="border-0 odd:bg-background even:bg-muted/30">
                <TableCell className="px-2 py-1 font-mono whitespace-nowrap">
                  {result.consumptionIndex}
                </TableCell>
                <TableCell className="px-2 py-1 font-mono whitespace-nowrap">
                  {formatTimer0Hex(result.timer0)}
                </TableCell>
                <TableCell className="px-2 py-1 font-mono whitespace-nowrap">
                  {formatVCountHex(result.vcount)}
                </TableCell>
                <TableCell className="px-2 py-1 font-mono whitespace-nowrap text-[10px]">
                  {result.initialSeed}
                </TableCell>
                <TableCell className="px-2 py-1 font-mono whitespace-nowrap text-[10px]">
                  {result.currentSeed}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    );
  };

  return (
    <PanelCard
      icon={<Compass size={20} className="opacity-80" />}
      title={<span>{reportNeedleCardTitle[locale]}</span>}
      headerActions={
        <div className="flex items-center gap-1">
          <Button
            type="button"
            size="sm"
            variant="ghost"
            onClick={handleCopyResults}
            disabled={results.length === 0}
            className="gap-1 h-7 px-2"
            title={reportNeedleButtonLabels.copy[locale]}
          >
            <Copy size={14} />
          </Button>
          <Button
            type="button"
            size="sm"
            variant="ghost"
            onClick={reset}
            disabled={isSearchActive}
            className="gap-1 h-7 px-2"
          >
            <ArrowCounterClockwise size={14} />
            {reportNeedleButtonLabels.reset[locale]}
          </Button>
        </div>
      }
      fullHeight={false}
      scrollMode="parent"
      spacing="compact"
    >
      {validationErrors.length > 0 && (
        <div className="text-destructive text-xs space-y-0.5" role="alert">
          {validationErrors.map((e, i) => (
            <div key={i}>{e}</div>
          ))}
        </div>
      )}

      {errorMessage && (
        <div className="text-destructive text-xs" role="alert">
          {errorMessage}
        </div>
      )}

      <div className="flex items-center gap-2 flex-wrap">
        {isIdle && (
          <Button size="sm" onClick={handleStart} className="flex-1">
            <Play size={16} className="mr-2" />
            {reportNeedleButtonLabels.start[locale]}
          </Button>
        )}
        {status === 'starting' && (
          <Button size="sm" disabled className="flex-1">
            <Play size={16} className="mr-2" />
            {reportNeedleButtonLabels.starting[locale]}
          </Button>
        )}
        {canPause && (
          <Button size="sm" variant="outline" onClick={pauseSearch} className="flex-1">
            <Pause size={16} className="mr-2" />
            {reportNeedleButtonLabels.pause[locale]}
          </Button>
        )}
        {canResume && (
          <Button size="sm" onClick={resumeSearch} className="flex-1">
            <Play size={16} className="mr-2" />
            {reportNeedleButtonLabels.resume[locale]}
          </Button>
        )}
        {(status === 'running' || status === 'paused') && (
          <Button size="sm" variant="destructive" onClick={stopSearch}>
            <Square size={16} className="mr-2" />
            {reportNeedleButtonLabels.stop[locale]}
          </Button>
        )}
        <div className="text-xs text-muted-foreground ml-auto">
          {reportNeedleStatusPrefix[locale]}: {statusDisplay}
          {' / '}
          {getReportNeedleModeLabel(draft.mode, locale)}
        </div>
      </div>

      {(status === 'running' || status === 'paused' || status === 'stopping') && (
        <div className="text-[11px] text-muted-foreground font-mono">検索中...</div>
      )}

      <div className="space-y-2">
        <Label className="text-xs">{reportNeedleParamLabels.mode[locale]}</Label>
        <ToggleGroup type="single" value={draft.mode} onValueChange={handleModeChange} className="flex flex-wrap gap-2">
          <ToggleGroupItem value="initial-seed" aria-label={getReportNeedleModeLabel('initial-seed', locale)}>
            {getReportNeedleModeLabel('initial-seed', locale)}
          </ToggleGroupItem>
          <ToggleGroupItem value="startup" aria-label={getReportNeedleModeLabel('startup', locale)}>
            {getReportNeedleModeLabel('startup', locale)}
          </ToggleGroupItem>
        </ToggleGroup>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label className="text-xs">{reportNeedleParamLabels.needleValue[locale]}</Label>
          <span className="text-[11px] text-muted-foreground">{reportNeedleParamLabels.needleHelper[locale]}</span>
        </div>
        <div className="flex items-center gap-2">
          <Input
            value={draft.needleValue}
            onChange={(e) => updateNeedleValue(e.target.value)}
            placeholder="0425"
            className="font-mono h-9"
            disabled={isSearchActive}
          />
          <div className="grid grid-cols-4 gap-1">
            {[0, 1, 2, 3, 4, 5, 6, 7].map((digit) => (
              <Button
                key={digit}
                type="button"
                size="icon"
                variant="secondary"
                className="h-8 w-8"
                disabled={isSearchActive}
                onClick={() => appendNeedleDigit(String(digit))}
              >
                {digit}
              </Button>
            ))}
          </div>
        </div>
        <div className="text-[11px] text-muted-foreground font-mono break-all">
          {combinedNeedle || '--'}
        </div>
      </div>

      <Separator />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="space-y-1">
          <Label className="text-xs">{reportNeedleParamLabels.timer0Range[locale]}</Label>
          <div className="flex items-center gap-2">
            <Input
              type="number"
              inputMode="numeric"
              value={draft.timer0Range.min}
              onChange={(e) => updateRange('timer0Range', 'min', parseInt(e.target.value, 10) || 0)}
              disabled={isSearchActive}
              className="h-8"
            />
            <span className="text-xs text-muted-foreground">~</span>
            <Input
              type="number"
              inputMode="numeric"
              value={draft.timer0Range.max}
              onChange={(e) => updateRange('timer0Range', 'max', parseInt(e.target.value, 10) || 0)}
              disabled={isSearchActive}
              className="h-8"
            />
          </div>
        </div>
        <div className="space-y-1">
          <Label className="text-xs">{reportNeedleParamLabels.vcountRange[locale]}</Label>
          <div className="flex items-center gap-2">
            <Input
              type="number"
              inputMode="numeric"
              value={draft.vcountRange.min}
              onChange={(e) => updateRange('vcountRange', 'min', parseInt(e.target.value, 10) || 0)}
              disabled={isSearchActive}
              className="h-8"
            />
            <span className="text-xs text-muted-foreground">~</span>
            <Input
              type="number"
              inputMode="numeric"
              value={draft.vcountRange.max}
              onChange={(e) => updateRange('vcountRange', 'max', parseInt(e.target.value, 10) || 0)}
              disabled={isSearchActive}
              className="h-8"
            />
          </div>
        </div>
      </div>

      <div className="space-y-1">
        <Label className="text-xs">{reportNeedleParamLabels.advanceRange[locale]}</Label>
        <div className="flex items-center gap-2">
          <Input
            type="number"
            inputMode="numeric"
            value={draft.advanceRange.start}
            onChange={(e) => updateAdvanceRange('start', parseInt(e.target.value, 10) || 0)}
            disabled={isSearchActive}
            className="h-9"
          />
          <span className="text-xs text-muted-foreground">~</span>
          <Input
            type="number"
            inputMode="numeric"
            value={draft.advanceRange.end}
            onChange={(e) => updateAdvanceRange('end', parseInt(e.target.value, 10) || 0)}
            disabled={isSearchActive}
            className="h-9"
          />
        </div>
      </div>

      <Separator />

      {draft.mode === 'startup' ? (
        <div className="space-y-3">
          <div className="space-y-1">
            <Label className="text-xs">{reportNeedleParamLabels.startDateTime[locale]}</Label>
            <div className="flex flex-col gap-2 min-[420px]:flex-row">
              <Input
                type="date"
                className="h-9 w-full min-w-[8rem]"
                min={DATE_INPUT_MIN}
                max={DATE_INPUT_MAX}
                disabled={isSearchActive}
                placeholder={reportNeedleStartupPlaceholders.bootDate[locale]}
                value={startupDateValue}
                onChange={(e) => handleStartupDateInput(e.target.value)}
              />
              <TimeInputHms
                idPrefix="report-startup-time"
                value={startupTimeValue}
                disabled={isSearchActive}
                onCommit={handleStartupTimeInput}
              />
            </div>
          </div>
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground" id="lbl-report-boot-keys" htmlFor="report-boot-keys-display">{reportNeedleParamLabels.keyInput[locale]}</Label>
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
              <div
                id="report-boot-keys-display"
                className="flex-1 min-h-[2.25rem] rounded-md border bg-muted/40 px-3 py-2 text-xs font-mono"
              >
                {keyDisplay}
              </div>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={handleOpenKeyDialog}
                disabled={isSearchActive}
                className="gap-2"
              >
                <GameController size={16} />
                {reportNeedleKeyLabels.configure[locale]}
              </Button>
            </div>
          </div>
          <KeyInputDialog
            isOpen={isKeyDialogOpen}
            onOpenChange={handleKeyDialogOpenChange}
            availableKeys={availableKeys}
            onToggleKey={handleToggleKey}
            onReset={handleResetKeys}
            onApply={handleApplyKeys}
            labels={{
              dialogTitle: reportNeedleKeyLabels.dialogTitle[locale],
              reset: reportNeedleKeyLabels.reset[locale],
              apply: reportNeedleKeyLabels.apply[locale],
            }}
            maxWidthClass="sm:max-w-lg"
          />
        </div>
      ) : (
        <div className="space-y-3">
          <div className="space-y-1">
            <Label className="text-xs">{reportNeedleParamLabels.initialSeed[locale]}</Label>
            <Input
              type="text"
              inputMode="text"
              placeholder="LCG Seed (hex)"
              value={draft.initialSeed.seedHex}
              onChange={(e) => setInitialSeedHex(e.target.value)}
              disabled={isSearchActive}
              className="font-mono h-9"
            />
          </div>
        </div>
      )}

      <Separator />

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label className="text-xs">{reportNeedleResultsLabel[locale]} ({results.length})</Label>
          <div className="text-[11px] text-muted-foreground font-mono">
            {combinedNeedle || '--'}
          </div>
        </div>
        {renderResultsTable()}
      </div>
    </PanelCard>
  );
};

import React from 'react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Toggle } from '@/components/ui/toggle';
import { GameController } from '@phosphor-icons/react';
import { useBootTimingDraft } from '@/hooks/generation/useBootTimingDraft';
import type { KeyName } from '@/lib/utils/key-input';

const SHOULDER_KEYS: KeyName[] = ['L', 'R'];
const FACE_KEYS: KeyName[] = ['X', 'Y', 'A', 'B'];
const START_SELECT_KEYS: KeyName[] = ['Select', 'Start'];
const DPAD_LAYOUT: Array<Array<KeyName | null>> = [
  [null, '[↑]', null],
  ['[←]', null, '[→]'],
  [null, '[↓]', null],
];
const KEY_ACCESSIBILITY_LABELS: Partial<Record<KeyName, string>> = {
  '[↑]': 'Up',
  '[↓]': 'Down',
  '[←]': 'Left',
  '[→]': 'Right',
};

export interface BootTimingLabels {
  timestamp: string;
  timestampPlaceholder: string;
  keyInput: string;
  profile: string;
  configure: string;
  dialogTitle: string;
  reset: string;
  apply: string;
}

interface BootTimingControlsProps {
  locale: 'ja' | 'en';
  disabled: boolean;
  isActive: boolean;
  labels: BootTimingLabels;
}

export const BootTimingControls: React.FC<BootTimingControlsProps> = ({ locale, disabled, isActive, labels }) => {
  const controller = useBootTimingDraft({ locale, disabled, isActive });
  const { snapshot, dialog, handleTimestampInput } = controller;

  const renderKeyToggle = React.useCallback((key: KeyName, elementKey?: string) => {
    const text = key.startsWith('[') && key.endsWith(']') ? key.slice(1, -1) : key;
    return (
      <Toggle
        key={elementKey ?? key}
        value={key}
        aria-label={KEY_ACCESSIBILITY_LABELS[key] ?? key}
        pressed={dialog.availableKeys.includes(key)}
        onPressedChange={() => dialog.toggleKey(key)}
        className="h-10 min-w-[3rem] px-3"
      >
        {text}
      </Toggle>
    );
  }, [dialog]);

  return (
    <>
      <div className="flex flex-col gap-1 min-w-0">
        <Label className="text-xs" htmlFor="boot-timestamp">{labels.timestamp}</Label>
        <Input
          id="boot-timestamp"
          type="datetime-local"
          step={1}
          className="h-9"
          disabled={disabled}
          value={snapshot.bootTimestampValue}
          onChange={e => handleTimestampInput(e.target.value)}
          placeholder={labels.timestampPlaceholder}
        />
      </div>
      <div className="flex flex-col gap-1 min-w-0 lg:col-span-3">
        <Label className="text-xs" id="lbl-boot-keys" htmlFor="boot-keys-display">{labels.keyInput}</Label>
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          <div
            id="boot-keys-display"
            className="flex-1 min-h-[2.25rem] rounded-md border bg-muted/40 px-3 py-2 text-xs font-mono"
          >
            {snapshot.keyDisplay.length > 0 ? snapshot.keyDisplay : '—'}
          </div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={dialog.open}
            disabled={!dialog.canOpen}
          >
            {labels.configure}
          </Button>
        </div>
      </div>
      <div className="flex flex-col gap-1 min-w-0 lg:col-span-3">
        <div className="text-xs font-medium text-muted-foreground flex items-center gap-2">
          <GameController size={14} className="opacity-70" />
          <span>{labels.profile}</span>
        </div>
        <div className="rounded-md border bg-muted/30 px-3 py-2 text-xs font-mono space-y-1">
          {snapshot.profileSummaryLines.map(line => (
            <div key={line}>{line}</div>
          ))}
        </div>
      </div>
      <Dialog open={dialog.isOpen} onOpenChange={dialog.onOpenChange}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>{labels.dialogTitle}</DialogTitle>
          </DialogHeader>
          <div className="space-y-6">
            <div className="flex justify-center gap-3 flex-wrap">
              {SHOULDER_KEYS.map((key, index) => renderKeyToggle(key, `shoulder-${index}`))}
            </div>
            <div className="grid grid-cols-3 gap-2 place-items-center">
              {DPAD_LAYOUT.flatMap((row, rowIndex) =>
                row.map((key, colIndex) =>
                  key ? (
                    renderKeyToggle(key, `dpad-${key}-${rowIndex}-${colIndex}`)
                  ) : (
                    <span key={`blank-${rowIndex}-${colIndex}`} className="w-12 h-10" />
                  ),
                ),
              )}
            </div>
            <div className="flex justify-center gap-3 flex-wrap">
              {START_SELECT_KEYS.map((key, index) => renderKeyToggle(key, `start-${index}`))}
            </div>
            <div className="flex justify-center gap-3 flex-wrap">
              {FACE_KEYS.map((key, index) => renderKeyToggle(key, `face-${index}`))}
            </div>
            <div className="flex items-center justify-between border-t pt-3">
              <Button type="button" variant="outline" size="sm" onClick={dialog.resetKeys}>
                {labels.reset}
              </Button>
              <Button type="button" size="sm" onClick={dialog.applyKeys}>
                {labels.apply}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
};

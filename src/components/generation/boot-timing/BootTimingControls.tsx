import React from 'react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { KeyInputDialog } from '@/components/keys';
import { GameController } from '@phosphor-icons/react';
import { useBootTimingDraft } from '@/hooks/generation/useBootTimingDraft';
import { TimeInputHms } from '@/components/ui/time-input-hms';
import { DATE_INPUT_MAX, DATE_INPUT_MIN } from '@/components/ui/date-input-constraints';

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
  disabled: boolean;
  isActive: boolean;
  labels: BootTimingLabels;
}

export const BootTimingControls: React.FC<BootTimingControlsProps> = ({ disabled, isActive, labels }) => {
  const controller = useBootTimingDraft({ disabled, isActive });
  const { snapshot, dialog, handleDateInput, handleTimeInput } = controller;

  return (
    <>
      <div className="flex flex-col gap-1 min-w-0">
        <Label className="text-xs" htmlFor="boot-date">{labels.timestamp}</Label>
        <div className="flex flex-col gap-2 min-[420px]:flex-row">
          <Input
            id="boot-date"
            type="date"
            className="h-9 w-1/2 min-w-[8rem]"
            min={DATE_INPUT_MIN}
            max={DATE_INPUT_MAX}
            disabled={disabled}
            placeholder={labels.timestampPlaceholder}
            value={snapshot.bootDateValue}
            onChange={e => handleDateInput(e.target.value)}
          />
          <TimeInputHms
            idPrefix="boot-time"
            value={snapshot.bootTimeValue}
            disabled={disabled}
            onCommit={handleTimeInput}
          />
        </div>
      </div>
      <div className="flex flex-col gap-1 min-w-0 lg:col-span-3">
        <Label className="text-xs text-muted-foreground" id="lbl-boot-keys" htmlFor="boot-keys-display">{labels.keyInput}</Label>
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
            className="gap-2"
          >
            <GameController size={16} />
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
      <KeyInputDialog
        isOpen={dialog.isOpen}
        onOpenChange={dialog.onOpenChange}
        availableKeys={dialog.availableKeys}
        onToggleKey={dialog.toggleKey}
        onReset={dialog.resetKeys}
        onApply={dialog.applyKeys}
        labels={{
          dialogTitle: labels.dialogTitle,
          reset: labels.reset,
          apply: labels.apply,
        }}
        maxWidthClass="sm:max-w-lg"
      />
    </>
  );
};

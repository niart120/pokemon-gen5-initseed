import React from 'react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { KeyInputDialog } from '@/components/keys';
import { GameController } from '@phosphor-icons/react';
import { useBootTimingDraft } from '@/hooks/generation/useBootTimingDraft';

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
  const { snapshot, dialog, handleTimestampInput } = controller;

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

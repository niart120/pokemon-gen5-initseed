/**
 * Egg Boot-Timing Controls
 * 
 * Boot-Timing input controls for Egg Generation Panel.
 * Uses EggBootTimingDraft hook for state management.
 */

import React from 'react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { KeyInputDialog } from '@/components/keys';
import { GameController } from '@phosphor-icons/react';
import { useEggBootTimingDraft } from '@/hooks/egg/useEggBootTimingDraft';

export interface EggBootTimingLabels {
  timestamp: string;
  timestampPlaceholder: string;
  keyInput: string;
  profile: string;
  configure: string;
  dialogTitle: string;
  reset: string;
  apply: string;
}

interface EggBootTimingControlsProps {
  disabled: boolean;
  isActive: boolean;
  labels: EggBootTimingLabels;
}

export const EggBootTimingControls: React.FC<EggBootTimingControlsProps> = ({ disabled, isActive, labels }) => {
  const controller = useEggBootTimingDraft({ disabled, isActive });
  const { snapshot, dialog, handleTimestampInput } = controller;

  return (
    <>
      <div className="flex flex-col gap-1 min-w-0">
        <Label className="text-xs" htmlFor="egg-boot-timestamp">{labels.timestamp}</Label>
        <Input
          id="egg-boot-timestamp"
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
        <Label className="text-xs" id="lbl-egg-boot-keys" htmlFor="egg-boot-keys-display">{labels.keyInput}</Label>
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          <div
            id="egg-boot-keys-display"
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

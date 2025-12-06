import React from 'react';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { BootTimingControls, type BootTimingLabels } from './BootTimingControls';

interface GenerationBootTimingSectionProps {
  disabled: boolean;
  labels: BootTimingLabels;
  minAdvanceLabel: string;
  maxAdvanceLabel: string;
  offsetHex?: string;
  maxAdvances?: number;
  onOffsetHexChange: (hex: string) => void;
  onMaxAdvancesChange: (value: number) => void;
}

export const GenerationBootTimingSection: React.FC<GenerationBootTimingSectionProps> = ({
  disabled,
  labels,
  minAdvanceLabel,
  maxAdvanceLabel,
  offsetHex,
  maxAdvances,
  onOffsetHexChange,
  onMaxAdvancesChange,
}) => {
  const offsetValue = parseInt(offsetHex ?? '0', 16) || 0;
  const maxAdvValue = maxAdvances ?? 0;

  return (
    <div className="space-y-3 rounded-md border bg-muted/20 p-3">
      <BootTimingControls disabled={disabled} isActive labels={labels} />
      <div className="grid gap-3 grid-cols-1 sm:grid-cols-2">
        <div className="flex flex-col gap-1 min-w-0">
          <Label className="text-xs" htmlFor="min-advance">{minAdvanceLabel}</Label>
          <Input
            id="min-advance"
            type="number"
            inputMode="numeric"
            className="h-9"
            disabled={disabled}
            value={offsetValue}
            onChange={e => {
              const v = Number(e.target.value);
              onOffsetHexChange((Number.isNaN(v) ? 0 : v).toString(16));
            }}
            onBlur={() => {
              const current = parseInt(offsetHex ?? '0', 16) || 0;
              const clamped = Math.max(0, current);
              onOffsetHexChange(clamped.toString(16));
            }}
            placeholder="0"
          />
        </div>
        <div className="flex flex-col gap-1">
          <Label className="text-xs" htmlFor="max-adv">{maxAdvanceLabel}</Label>
          <Input
            id="max-adv"
            type="number"
            inputMode="numeric"
            className="h-9"
            disabled={disabled}
            value={maxAdvValue}
            onChange={e => {
              const v = Number(e.target.value);
              onMaxAdvancesChange(Number.isNaN(v) ? 0 : v);
            }}
            onBlur={() => {
              const current = maxAdvValue ?? 0;
              const clamped = Math.max(0, current);
              onMaxAdvancesChange(clamped);
            }}
          />
        </div>
      </div>
    </div>
  );
};

import React from 'react';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { EggBootTimingControls, type EggBootTimingLabels } from './EggBootTimingControls';

interface EggBootTimingSectionProps {
  disabled: boolean;
  labels: EggBootTimingLabels;
  userOffsetLabel: string;
  countLabel: string;
  userOffsetHex: string;
  count: number;
  onUserOffsetHexChange: (hex: string) => void;
  onCountChange: (value: number) => void;
}

export const EggBootTimingSection: React.FC<EggBootTimingSectionProps> = ({
  disabled,
  labels,
  userOffsetLabel,
  countLabel,
  userOffsetHex,
  count,
  onUserOffsetHexChange,
  onCountChange,
}) => {
  const offsetValue = parseInt(userOffsetHex, 16) || 0;
  const countValue = count || 0;

  return (
    <div className="space-y-3 rounded-md border bg-muted/20 p-3">
      <EggBootTimingControls disabled={disabled} isActive labels={labels} />
      <div className="grid gap-3 grid-cols-1 sm:grid-cols-2">
        <div className="flex flex-col gap-1 min-w-0">
          <Label className="text-xs" htmlFor="egg-user-offset-bt">{userOffsetLabel}</Label>
          <Input
            id="egg-user-offset-bt"
            data-testid="egg-user-offset-bt"
            type="number"
            inputMode="numeric"
            min={0}
            value={offsetValue}
            onChange={(e) => {
              onUserOffsetHexChange(e.target.value);
            }}
            onBlur={(e) => {
              const num = Math.max(0, parseInt(e.target.value) || 0);
              onUserOffsetHexChange(num.toString(16).toUpperCase());
            }}
            disabled={disabled}
            className="h-9"
          />
        </div>
        <div className="flex flex-col gap-1 min-w-0">
          <Label className="text-xs" htmlFor="egg-count-bt">{countLabel}</Label>
          <Input
            id="egg-count-bt"
            data-testid="egg-count-bt"
            type="number"
            inputMode="numeric"
            min={1}
            max={100000}
            value={countValue}
            onChange={(e) => {
              const v = parseInt(e.target.value, 10);
              onCountChange(Number.isNaN(v) ? 0 : v);
            }}
            onBlur={() => {
              const num = Math.max(1, Math.min(100000, countValue || 1));
              onCountChange(num);
            }}
            disabled={disabled}
            className="h-9"
          />
        </div>
      </div>
    </div>
  );
};

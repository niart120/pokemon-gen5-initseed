import React from 'react';
import { Switch } from '@/components/ui/switch';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useLocale } from '@/lib/i18n/locale-context';
import {
  resolveProfileTimerAutoAria,
  resolveProfileTimerAutoLabel,
  resolveProfileTimerFieldLabel,
} from '@/lib/i18n/strings/profile-timer';

type TimerRangeField = 'timer0Min' | 'timer0Max';
type VCountRangeField = 'vcountMin' | 'vcountMax';

interface Timer0VCountSectionProps {
  timer0Auto: boolean;
  timer0Min: string;
  timer0Max: string;
  vcountMin: string;
  vcountMax: string;
  onAutoToggle: (checked: boolean) => void;
  onTimerHexChange: (field: TimerRangeField, value: string) => void;
  onTimerHexBlur: (field: TimerRangeField) => void;
  onVCountHexChange: (field: VCountRangeField, value: string) => void;
  onVCountHexBlur: (field: VCountRangeField) => void;
  disabled?: boolean;
}

export function Timer0VCountSection({
  timer0Auto,
  timer0Min,
  timer0Max,
  vcountMin,
  vcountMax,
  onAutoToggle,
  onTimerHexChange,
  onTimerHexBlur,
  onVCountHexChange,
  onVCountHexBlur,
  disabled = false,
}: Timer0VCountSectionProps) {
  const locale = useLocale();

  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-3 lg:grid-cols-5">
      <div className="flex min-w-0 flex-col gap-1">
        <Label htmlFor="timer0-min" className="text-xs">
          {resolveProfileTimerFieldLabel('timer0Min', locale)}
        </Label>
        <Input
          id="timer0-min"
          value={timer0Min}
          onChange={(event) => onTimerHexChange('timer0Min', event.target.value)}
          onBlur={() => onTimerHexBlur('timer0Min')}
          disabled={timer0Auto || disabled}
          className="h-9 w-full min-w-0 px-2 font-mono text-xs"
          placeholder="0x0"
        />
      </div>
      <div className="flex min-w-0 flex-col gap-1">
        <Label htmlFor="timer0-max" className="text-xs">
          {resolveProfileTimerFieldLabel('timer0Max', locale)}
        </Label>
        <Input
          id="timer0-max"
          value={timer0Max}
          onChange={(event) => onTimerHexChange('timer0Max', event.target.value)}
          onBlur={() => onTimerHexBlur('timer0Max')}
          disabled={timer0Auto || disabled}
          className="h-9 w-full min-w-0 px-2 font-mono text-xs"
          placeholder="0x0"
        />
      </div>
      <div className="flex min-w-0 flex-col gap-1">
        <Label htmlFor="vcount-min" className="text-xs">
          {resolveProfileTimerFieldLabel('vcountMin', locale)}
        </Label>
        <Input
          id="vcount-min"
          value={vcountMin}
          onChange={(event) => onVCountHexChange('vcountMin', event.target.value)}
          onBlur={() => onVCountHexBlur('vcountMin')}
          disabled={timer0Auto || disabled}
          className="h-9 w-full min-w-0 px-2 font-mono text-xs"
          placeholder="0x0"
        />
      </div>
      <div className="flex min-w-0 flex-col gap-1">
        <Label htmlFor="vcount-max" className="text-xs">
          {resolveProfileTimerFieldLabel('vcountMax', locale)}
        </Label>
        <Input
          id="vcount-max"
          value={vcountMax}
          onChange={(event) => onVCountHexChange('vcountMax', event.target.value)}
          onBlur={() => onVCountHexBlur('vcountMax')}
          disabled={timer0Auto || disabled}
          className="h-9 w-full min-w-0 px-2 font-mono text-xs"
          placeholder="0x0"
        />
      </div>
      <div className="flex min-w-0 flex-col gap-1">
        <Label htmlFor="profile-timer-auto" className="text-xs">
          {resolveProfileTimerAutoLabel(locale)}
        </Label>
        <Switch
          id="profile-timer-auto"
          checked={timer0Auto}
          onCheckedChange={(value) => onAutoToggle(Boolean(value))}
          disabled={disabled}
          aria-label={resolveProfileTimerAutoAria(locale)}
        />
      </div>
    </div>
  );
}

import React from 'react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

type TimerRangeField = 'timer0Min' | 'timer0Max';
type VCountRangeField = 'vcountMin' | 'vcountMax';

interface Timer0VCountSectionProps {
  timer0Auto: boolean;
  timer0Min: string;
  timer0Max: string;
  vcountMin: string;
  vcountMax: string;
  onTimerHexChange: (field: TimerRangeField, value: string) => void;
  onTimerHexBlur: (field: TimerRangeField) => void;
  onVCountHexChange: (field: VCountRangeField, value: string) => void;
  onVCountHexBlur: (field: VCountRangeField) => void;
}

export function Timer0VCountSection({
  timer0Auto,
  timer0Min,
  timer0Max,
  vcountMin,
  vcountMax,
  onTimerHexChange,
  onTimerHexBlur,
  onVCountHexChange,
  onVCountHexBlur,
}: Timer0VCountSectionProps) {
  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-4">
      <div className="flex min-w-0 flex-col gap-1">
        <Label htmlFor="timer0-min" className="text-xs">Timer0 Min</Label>
        <Input
          id="timer0-min"
          value={timer0Min}
          onChange={(event) => onTimerHexChange('timer0Min', event.target.value)}
          onBlur={() => onTimerHexBlur('timer0Min')}
          disabled={timer0Auto}
          className="h-9 w-full min-w-0 px-2 font-mono text-xs"
          placeholder="0x0"
        />
      </div>
      <div className="flex min-w-0 flex-col gap-1">
        <Label htmlFor="timer0-max" className="text-xs">Timer0 Max</Label>
        <Input
          id="timer0-max"
          value={timer0Max}
          onChange={(event) => onTimerHexChange('timer0Max', event.target.value)}
          onBlur={() => onTimerHexBlur('timer0Max')}
          disabled={timer0Auto}
          className="h-9 w-full min-w-0 px-2 font-mono text-xs"
          placeholder="0x0"
        />
      </div>
      <div className="flex min-w-0 flex-col gap-1">
        <Label htmlFor="vcount-min" className="text-xs">VCount Min</Label>
        <Input
          id="vcount-min"
          value={vcountMin}
          onChange={(event) => onVCountHexChange('vcountMin', event.target.value)}
          onBlur={() => onVCountHexBlur('vcountMin')}
          disabled={timer0Auto}
          className="h-9 w-full min-w-0 px-2 font-mono text-xs"
          placeholder="0x0"
        />
      </div>
      <div className="flex min-w-0 flex-col gap-1">
        <Label htmlFor="vcount-max" className="text-xs">VCount Max</Label>
        <Input
          id="vcount-max"
          value={vcountMax}
          onChange={(event) => onVCountHexChange('vcountMax', event.target.value)}
          onBlur={() => onVCountHexBlur('vcountMax')}
          disabled={timer0Auto}
          className="h-9 w-full min-w-0 px-2 font-mono text-xs"
          placeholder="0x0"
        />
      </div>
    </div>
  );
}

import React from 'react';
import { Input } from '@/components/ui/input';

interface TimeInputHmsProps {
  value: string;            // expected HH:MM:SS (zero-padded) – fallback handled internally
  disabled?: boolean;
  onCommit: (next: string) => void; // called on blur/Enter with normalized HH:MM:SS
  idPrefix?: string;
}

export const TimeInputHms: React.FC<TimeInputHmsProps> = ({ value, disabled, onCommit, idPrefix }) => {
  const [parts, setParts] = React.useState(splitToParts(value));

  React.useEffect(() => {
    setParts(splitToParts(value));
  }, [value]);

  const commit = React.useCallback(() => {
    const normalized = normalizeParts(parts);
    setParts(normalized);
    onCommit(compose(normalized));
  }, [onCommit, parts]);

  const handleChange = (key: keyof typeof parts) => (e: React.ChangeEvent<HTMLInputElement>) => {
    const digits = e.target.value.replace(/\D/g, '').slice(0, 2);
    setParts(prev => ({ ...prev, [key]: digits }));
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      commit();
    }
  };

  const hourId = idPrefix ? `${idPrefix}-hour` : undefined;
  const minuteId = idPrefix ? `${idPrefix}-minute` : undefined;
  const secondId = idPrefix ? `${idPrefix}-second` : undefined;

  return (
    <div className="flex items-center gap-1">
      <Input
        id={hourId}
        inputMode="numeric"
        pattern="\\d*"
        maxLength={2}
        className="h-9 w-14 text-center font-mono"
        disabled={disabled}
        value={parts.hour}
        onChange={handleChange('hour')}
        onBlur={commit}
        onKeyDown={handleKeyDown}
        placeholder="00"
      />
      <span className="text-sm text-muted-foreground">:</span>
      <Input
        id={minuteId}
        inputMode="numeric"
        pattern="\\d*"
        maxLength={2}
        className="h-9 w-14 text-center font-mono"
        disabled={disabled}
        value={parts.minute}
        onChange={handleChange('minute')}
        onBlur={commit}
        onKeyDown={handleKeyDown}
        placeholder="00"
      />
      <span className="text-sm text-muted-foreground">:</span>
      <Input
        id={secondId}
        inputMode="numeric"
        pattern="\\d*"
        maxLength={2}
        className="h-9 w-14 text-center font-mono"
        disabled={disabled}
        value={parts.second}
        onChange={handleChange('second')}
        onBlur={commit}
        onKeyDown={handleKeyDown}
        placeholder="00"
      />
    </div>
  );
};

function splitToParts(value: string): { hour: string; minute: string; second: string } {
  const [h = '', m = '', s = ''] = value?.split(':') ?? [];
  return {
    hour: h.padStart(2, '0').slice(0, 2),
    minute: m.padStart(2, '0').slice(0, 2),
    second: s.padStart(2, '0').slice(0, 2),
  };
}

function normalizeParts(parts: { hour: string; minute: string; second: string }) {
  return {
    hour: clamp(parts.hour, 23),
    minute: clamp(parts.minute, 59),
    second: clamp(parts.second, 59),
  };
}

function clamp(raw: string, max: number): string {
  const digits = raw.replace(/\D/g, '');
  if (digits === '') return '00';
  const num = Math.min(max, Number(digits));
  return num.toString().padStart(2, '0');
}

function compose(parts: { hour: string; minute: string; second: string }): string {
  return `${padOrDefault(parts.hour)}:${padOrDefault(parts.minute)}:${padOrDefault(parts.second)}`;
}

function padOrDefault(v: string): string {
  if (!v) return '00';
  const digits = v.replace(/\D/g, '').slice(0, 2);
  if (digits === '') return '00';
  return digits.padStart(2, '0');
}

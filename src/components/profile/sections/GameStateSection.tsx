import React from 'react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveProfileGameFieldLabel } from '@/lib/i18n/strings/profile-game';

interface GameStateSectionProps {
  tid: string;
  sid: string;
  newGame: boolean;
  withSave: boolean;
  shinyCharm: boolean;
  memoryLink: boolean;
  onTidChange: (value: string) => void;
  onTidBlur: () => void;
  onSidChange: (value: string) => void;
  onSidBlur: () => void;
  onNewGameToggle: (checked: boolean) => void;
  onWithSaveToggle: (checked: boolean) => void;
  onShinyCharmToggle: (checked: boolean) => void;
  onMemoryLinkToggle: (checked: boolean) => void;
  withSaveDisabled: boolean;
  shinyCharmDisabled: boolean;
  memoryLinkDisabled: boolean;
  disabled?: boolean;
}

export function GameStateSection({
  tid,
  sid,
  newGame,
  withSave,
  shinyCharm,
  memoryLink,
  onTidChange,
  onTidBlur,
  onSidChange,
  onSidBlur,
  onNewGameToggle,
  onWithSaveToggle,
  onShinyCharmToggle,
  onMemoryLinkToggle,
  withSaveDisabled,
  shinyCharmDisabled,
  memoryLinkDisabled,
  disabled = false,
}: GameStateSectionProps) {
  const locale = useLocale();

  return (
    <div className="grid items-end gap-3 sm:grid-cols-2 md:grid-cols-4 lg:grid-cols-6">
      <div className="flex flex-col gap-1">
        <Label htmlFor="profile-tid" className="text-xs">
          {resolveProfileGameFieldLabel('tid', locale)}
        </Label>
        <Input
          id="profile-tid"
          value={tid}
          onChange={(event) => onTidChange(event.target.value)}
          onBlur={onTidBlur}
          placeholder="00000"
          inputMode="numeric"
          className="h-9"
          disabled={disabled}
        />
      </div>
      <div className="flex flex-col gap-1">
        <Label htmlFor="profile-sid" className="text-xs">
          {resolveProfileGameFieldLabel('sid', locale)}
        </Label>
        <Input
          id="profile-sid"
          value={sid}
          onChange={(event) => onSidChange(event.target.value)}
          onBlur={onSidBlur}
          placeholder="00000"
          inputMode="numeric"
          className="h-9"
          disabled={disabled}
        />
      </div>
      <ToggleField
        id="profile-new-game"
        label={resolveProfileGameFieldLabel('newGame', locale)}
        checked={newGame}
        onChange={onNewGameToggle}
        disabled={disabled}
      />
      <ToggleField
        id="profile-with-save"
        label={resolveProfileGameFieldLabel('withSave', locale)}
        checked={withSave}
        onChange={onWithSaveToggle}
        disabled={withSaveDisabled || disabled}
      />
      <ToggleField
        id="profile-shiny"
        label={resolveProfileGameFieldLabel('shinyCharm', locale)}
        checked={shinyCharm}
        onChange={onShinyCharmToggle}
        disabled={shinyCharmDisabled || disabled}
      />
      <ToggleField
        id="profile-memory-link"
        label={resolveProfileGameFieldLabel('memoryLink', locale)}
        checked={memoryLink}
        onChange={onMemoryLinkToggle}
        disabled={memoryLinkDisabled || disabled}
      />
    </div>
  );
}

interface ToggleFieldProps {
  id: string;
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}

function ToggleField({ id, label, checked, onChange, disabled = false }: ToggleFieldProps) {
  const labelClasses = `text-xs${disabled ? ' text-muted-foreground' : ''}`;

  return (
    <div className="flex min-w-0 flex-col gap-1">
      <Label htmlFor={id} className={labelClasses}>{label}</Label>
      <Switch
        id={id}
        checked={checked}
        onCheckedChange={(value) => onChange(Boolean(value))}
        disabled={disabled}
        aria-label={label}
      />
    </div>
  );
}

import React from 'react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';

interface GameStateSectionProps {
  tid: string;
  sid: string;
  newGame: boolean;
  withSave: boolean;
  shinyCharm: boolean;
  memoryLink: boolean;
  onTidChange: (value: string) => void;
  onSidChange: (value: string) => void;
  onNewGameToggle: (checked: boolean) => void;
  onWithSaveToggle: (checked: boolean) => void;
  onShinyCharmToggle: (checked: boolean) => void;
  onMemoryLinkToggle: (checked: boolean) => void;
  withSaveDisabled: boolean;
  shinyCharmDisabled: boolean;
  memoryLinkDisabled: boolean;
}

export function GameStateSection({
  tid,
  sid,
  newGame,
  withSave,
  shinyCharm,
  memoryLink,
  onTidChange,
  onSidChange,
  onNewGameToggle,
  onWithSaveToggle,
  onShinyCharmToggle,
  onMemoryLinkToggle,
  withSaveDisabled,
  shinyCharmDisabled,
  memoryLinkDisabled,
}: GameStateSectionProps) {
  return (
    <div className="grid items-end gap-3 sm:grid-cols-2 md:grid-cols-4 lg:grid-cols-6">
      <div className="flex flex-col gap-1">
        <Label htmlFor="profile-tid" className="text-xs">TID</Label>
        <Input
          id="profile-tid"
          value={tid}
          onChange={(event) => onTidChange(event.target.value)}
          placeholder="00000"
          inputMode="numeric"
          className="h-9"
        />
      </div>
      <div className="flex flex-col gap-1">
        <Label htmlFor="profile-sid" className="text-xs">SID</Label>
        <Input
          id="profile-sid"
          value={sid}
          onChange={(event) => onSidChange(event.target.value)}
          placeholder="00000"
          inputMode="numeric"
          className="h-9"
        />
      </div>
      <ToggleField
        id="profile-new-game"
        label="New Game"
        checked={newGame}
        onChange={onNewGameToggle}
      />
      <ToggleField
        id="profile-with-save"
        label="With Save"
        checked={withSave}
        onChange={onWithSaveToggle}
        disabled={withSaveDisabled}
      />
      <ToggleField
        id="profile-shiny"
        label="Shiny Charm"
        checked={shinyCharm}
        onChange={onShinyCharmToggle}
        disabled={shinyCharmDisabled}
      />
      <ToggleField
        id="profile-memory-link"
        label="Memory Link"
        checked={memoryLink}
        onChange={onMemoryLinkToggle}
        disabled={memoryLinkDisabled}
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

import React from 'react';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

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
      <div className="flex items-center gap-2">
        <Checkbox
          id="profile-new-game"
          checked={newGame}
          onCheckedChange={(value) => onNewGameToggle(Boolean(value))}
        />
        <Label htmlFor="profile-new-game" className="text-xs">New Game</Label>
      </div>
      <div className="flex items-center gap-2">
        <Checkbox
          id="profile-with-save"
          checked={withSave}
          onCheckedChange={(value) => onWithSaveToggle(Boolean(value))}
          disabled={withSaveDisabled}
        />
        <Label htmlFor="profile-with-save" className="text-xs">With Save</Label>
      </div>
      <div className="flex items-center gap-2">
        <Checkbox
          id="profile-shiny"
          checked={shinyCharm}
          onCheckedChange={(value) => onShinyCharmToggle(Boolean(value))}
          disabled={shinyCharmDisabled}
        />
        <Label htmlFor="profile-shiny" className="text-xs">Shiny Charm</Label>
      </div>
      <div className="flex items-center gap-2">
        <Checkbox
          id="profile-memory-link"
          checked={memoryLink}
          onCheckedChange={(value) => onMemoryLinkToggle(Boolean(value))}
          disabled={memoryLinkDisabled}
        />
        <Label htmlFor="profile-memory-link" className="text-xs">Memory Link</Label>
      </div>
    </div>
  );
}

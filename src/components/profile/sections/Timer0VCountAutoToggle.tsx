import React from 'react';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';

interface Timer0VCountAutoToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
}

export function Timer0VCountAutoToggle({ checked, onChange }: Timer0VCountAutoToggleProps) {
  return (
    <div className="flex items-center gap-2">
      <Checkbox
        id="profile-timer-auto"
        checked={checked}
        onCheckedChange={(value) => onChange(Boolean(value))}
      />
      <Label htmlFor="profile-timer-auto" className="text-xs">Auto</Label>
    </div>
  );
}

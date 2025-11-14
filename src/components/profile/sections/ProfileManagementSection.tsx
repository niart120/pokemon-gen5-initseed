import React from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import type { DeviceProfile } from '@/types/profile';
import { SELECT_IMPORT_CURRENT, SELECT_NEW_PROFILE } from '@/components/profile/hooks/useProfileCardForm';

interface ProfileManagementSectionProps {
  profiles: DeviceProfile[];
  activeProfileId: string;
  profileName: string;
  dirty: boolean;
  canModify: boolean;
  disableDelete: boolean;
  onSelectProfile: (profileId: string) => void;
  onProfileNameChange: (value: string) => void;
  onSave: () => void;
  onLoad: () => void;
  onDelete: () => void;
}

export function ProfileManagementSection({
  profiles,
  activeProfileId,
  profileName,
  dirty,
  canModify,
  disableDelete,
  onSelectProfile,
  onProfileNameChange,
  onSave,
  onLoad,
  onDelete,
}: ProfileManagementSectionProps) {
  return (
    <div className="flex flex-wrap items-end gap-3">
      <div className="flex min-w-[12rem] flex-col gap-1 sm:w-56">
        <Label htmlFor="profile-select" className="text-xs">Profile</Label>
        <Select value={activeProfileId} onValueChange={onSelectProfile}>
          <SelectTrigger id="profile-select" className="h-9">
            <SelectValue placeholder="プロファイルを選択" />
          </SelectTrigger>
          <SelectContent>
            {profiles.map((profile) => (
              <SelectItem key={profile.id} value={profile.id}>
                {profile.name}
              </SelectItem>
            ))}
            <div className="my-1 h-px bg-border" role="separator" />
            <SelectItem value={SELECT_NEW_PROFILE}>+ New profile</SelectItem>
            <SelectItem value={SELECT_IMPORT_CURRENT}>Import current settings</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className="flex min-w-[12rem] flex-1 flex-col gap-1 sm:w-64">
        <Label htmlFor="profile-name" className="text-xs">Profile Name</Label>
        <Input
          id="profile-name"
          value={profileName}
          onChange={(event) => onProfileNameChange(event.target.value)}
          placeholder="My profile"
          className="h-9"
        />
      </div>
      <div className="ml-auto flex items-center gap-2">
        {dirty && <Badge variant="secondary">未保存</Badge>}
        <Button size="sm" variant="outline" onClick={onSave} disabled={!canModify}>Save</Button>
        <Button size="sm" variant="outline" onClick={onLoad} disabled={!canModify}>Load</Button>
        <Button size="sm" variant="destructive" onClick={onDelete} disabled={disableDelete}>Delete</Button>
      </div>
    </div>
  );
}

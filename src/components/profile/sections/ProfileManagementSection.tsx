import React from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { ProfileRenameDialog } from '@/components/profile/sections/ProfileRenameDialog';
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
  const [renameOpen, setRenameOpen] = React.useState(false);

  React.useEffect(() => {
    if (!canModify && renameOpen) {
      setRenameOpen(false);
    }
  }, [canModify, renameOpen]);

  const handleRenameOpenChange = React.useCallback(
    (nextOpen: boolean) => {
      if (nextOpen && !canModify) {
        return;
      }
      setRenameOpen(nextOpen);
    },
    [canModify],
  );

  const onRename = () =>  handleRenameOpenChange(true);

  return (
    <div className="flex flex-wrap items-end gap-3">
      <div className="flex flex-col gap-1">
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
        {dirty && <Badge variant="secondary">未保存</Badge>}
        <Button size="sm" variant="outline" onClick={onRename} disabled={!canModify}>Rename</Button>
        <Button size="sm" variant="outline" onClick={onSave} disabled={!canModify}>Save</Button>
        <Button size="sm" variant="outline" onClick={onLoad} disabled={!canModify}>Load</Button>
        <Button size="sm" variant="destructive" onClick={onDelete} disabled={disableDelete}>Delete</Button>
      <ProfileRenameDialog
        profileName={profileName}
        open={renameOpen}
        onOpenChange={handleRenameOpenChange}
        onRename={onProfileNameChange}
      />
    </div>
  );
}

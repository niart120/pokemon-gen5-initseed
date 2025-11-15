import React from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { ProfileRenameDialog } from '@/components/profile/sections/ProfileRenameDialog';
import type { DeviceProfile } from '@/types/profile';
import { SELECT_IMPORT_CURRENT, SELECT_NEW_PROFILE } from '@/components/profile/hooks/useProfileCardForm';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import {
  profileManagementSelectLabel,
  profileManagementSelectPlaceholder,
  profileManagementNewProfileLabel,
  profileManagementImportCurrentLabel,
  profileManagementDirtyBadge,
  resolveProfileManagementButtonLabel,
} from '@/lib/i18n/strings/profile-management';

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
  onDelete,
}: ProfileManagementSectionProps) {
  const [renameOpen, setRenameOpen] = React.useState(false);
  const locale = useLocale();

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

  const onRename = () => handleRenameOpenChange(true);

  return (
    <div className="flex flex-wrap items-end gap-3">
      <div className="flex flex-col gap-1">
        <Label htmlFor="profile-select" className="text-xs">
          {resolveLocaleValue(profileManagementSelectLabel, locale)}
        </Label>
        <Select value={activeProfileId} onValueChange={onSelectProfile}>
          <SelectTrigger id="profile-select" className="h-9">
            <SelectValue placeholder={resolveLocaleValue(profileManagementSelectPlaceholder, locale)} />
          </SelectTrigger>
          <SelectContent>
            {profiles.map((profile) => (
              <SelectItem key={profile.id} value={profile.id}>
                {profile.name}
              </SelectItem>
            ))}
            <div className="my-1 h-px bg-border" role="separator" />
            <SelectItem value={SELECT_NEW_PROFILE}>
              {resolveLocaleValue(profileManagementNewProfileLabel, locale)}
            </SelectItem>
            <SelectItem value={SELECT_IMPORT_CURRENT}>
              {resolveLocaleValue(profileManagementImportCurrentLabel, locale)}
            </SelectItem>
          </SelectContent>
        </Select>
      </div>
      <Button size="sm" variant="outline" onClick={onRename} disabled={!canModify}>
        {resolveProfileManagementButtonLabel('rename', locale)}
      </Button>
      <Button size="sm" variant="outline" onClick={onSave} disabled={!canModify}>
        {resolveProfileManagementButtonLabel('save', locale)}
      </Button>
      <Button size="sm" variant="destructive" onClick={onDelete} disabled={disableDelete}>
        {resolveProfileManagementButtonLabel('delete', locale)}
      </Button>
      <ProfileRenameDialog
        profileName={profileName}
        open={renameOpen}
        onOpenChange={handleRenameOpenChange}
        onRename={onProfileNameChange}
      />
      
      {dirty && (
        <Badge variant="secondary">
          {resolveLocaleValue(profileManagementDirtyBadge, locale)}
        </Badge>
      )}
    </div>
  );
}

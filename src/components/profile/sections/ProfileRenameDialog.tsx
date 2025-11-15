import React from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { useLocale } from '@/lib/i18n/locale-context';
import {
  profileRenameCancelButton,
  profileRenameDialogDescription,
  profileRenameDialogTitle,
  profileRenameFieldLabel,
  profileRenameFieldPlaceholder,
  profileRenameSubmitButton,
  resolveProfileRenameValue,
} from '@/lib/i18n/strings/profile-rename';

interface ProfileRenameDialogProps {
  profileName: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onRename: (value: string) => void;
}

export function ProfileRenameDialog({ profileName, open, onOpenChange, onRename }: ProfileRenameDialogProps) {
  const [value, setValue] = React.useState(profileName);
  const locale = useLocale();

  React.useEffect(() => {
    if (open) {
      setValue(profileName);
    }
  }, [open, profileName]);

  const handleSubmit = () => {
    const nextName = value.trim();
    if (!nextName) {
      return;
    }
    if (nextName !== profileName) {
      onRename(nextName);
    }
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle>{resolveProfileRenameValue(profileRenameDialogTitle, locale)}</DialogTitle>
          <DialogDescription>{resolveProfileRenameValue(profileRenameDialogDescription, locale)}</DialogDescription>
        </DialogHeader>
        <form
          className="space-y-4"
          onSubmit={(event) => {
            event.preventDefault();
            handleSubmit();
          }}
        >
          <div className="flex flex-col gap-1">
            <Label htmlFor="profile-rename" className="text-xs uppercase tracking-wide text-muted-foreground">
              {resolveProfileRenameValue(profileRenameFieldLabel, locale)}
            </Label>
            <Input
              id="profile-rename"
              value={value}
              onChange={(event) => setValue(event.target.value)}
              placeholder={resolveProfileRenameValue(profileRenameFieldPlaceholder, locale)}
              autoFocus
            />
          </div>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
              {resolveProfileRenameValue(profileRenameCancelButton, locale)}
            </Button>
            <Button type="submit" disabled={!value.trim()}>
              {resolveProfileRenameValue(profileRenameSubmitButton, locale)}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

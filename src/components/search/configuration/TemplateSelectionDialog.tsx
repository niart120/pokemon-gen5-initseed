import React from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { ScrollArea } from '@/components/ui/scroll-area';
import { SEED_TEMPLATES, type SeedTemplate } from '@/data/seed-templates';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import {
  formatTemplateSelectionApplyButtonLabel,
  formatTemplateSelectionSeedCount,
  templateSelectionCancelButtonLabel,
  templateSelectionDialogDescription,
  templateSelectionDialogTitle,
} from '@/lib/i18n/strings/search-template-selection';

interface TemplateSelectionDialogProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  onApplyTemplate: (seeds: number[]) => void;
}

export function TemplateSelectionDialog({
  isOpen,
  onOpenChange,
  onApplyTemplate,
}: TemplateSelectionDialogProps) {
  const [selectedTemplates, setSelectedTemplates] = React.useState<Set<string>>(new Set());
  const locale = useLocale();

  const handleToggleTemplate = (templateName: string) => {
    const newSelection = new Set(selectedTemplates);
    if (newSelection.has(templateName)) {
      newSelection.delete(templateName);
    } else {
      newSelection.add(templateName);
    }
    setSelectedTemplates(newSelection);
  };

  const handleApply = () => {
    // Collect all seeds from selected templates
    const allSeeds = new Set<number>();
    SEED_TEMPLATES.forEach(template => {
      if (selectedTemplates.has(template.name)) {
        template.seeds.forEach(seed => allSeeds.add(seed));
      }
    });

    // Convert to array and apply
    onApplyTemplate(Array.from(allSeeds));
    onOpenChange(false);
    
    // Reset selection for next time
    setSelectedTemplates(new Set());
  };

  const handleCancel = () => {
    setSelectedTemplates(new Set());
    onOpenChange(false);
  };

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[80vh]">
        <DialogHeader>
          <DialogTitle>{resolveLocaleValue(templateSelectionDialogTitle, locale)}</DialogTitle>
        </DialogHeader>
        
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground">
            {resolveLocaleValue(templateSelectionDialogDescription, locale)}
          </p>

          <ScrollArea className="h-[400px] pr-4">
            <div className="space-y-3">
              {SEED_TEMPLATES.map((template: SeedTemplate) => (
                <div
                  key={template.name}
                  className="flex items-start space-x-3 p-3 border rounded-lg hover:bg-accent/50 cursor-pointer"
                  onClick={() => handleToggleTemplate(template.name)}
                >
                  <Checkbox
                    id={`template-${template.name}`}
                    checked={selectedTemplates.has(template.name)}
                    onCheckedChange={() => handleToggleTemplate(template.name)}
                    onClick={(e) => e.stopPropagation()}
                  />
                  <div className="flex-1 space-y-1">
                    <Label
                      htmlFor={`template-${template.name}`}
                      className="cursor-pointer font-medium"
                    >
                      {template.name}
                    </Label>
                    {template.description && (
                      <p className="text-sm text-muted-foreground">
                        {template.description}
                      </p>
                    )}
                    <p className="text-xs text-muted-foreground">
                      {formatTemplateSelectionSeedCount(template.seeds.length, locale)}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>

          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={handleCancel}>
              {resolveLocaleValue(templateSelectionCancelButtonLabel, locale)}
            </Button>
            <Button 
              onClick={handleApply}
              disabled={selectedTemplates.size === 0}
            >
              {formatTemplateSelectionApplyButtonLabel(selectedTemplates.size, locale)}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

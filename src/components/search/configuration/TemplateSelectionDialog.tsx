import React from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { SEED_TEMPLATES, type SeedTemplate, type TemplateVersion } from '@/data/seed-templates';
import { useAppStore } from '../../../store/app-store';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import {
  formatTemplateSelectionApplyButtonLabel,
  formatTemplateSelectionSeedCount,
  getTemplateCategoryLabel,
  templateCategoryFilterLabel,
  templateCategoryOptions,
  templateNoResultsMessage,
  templateSelectionCancelButtonLabel,
  templateSelectionDialogDescription,
  templateSelectionDialogTitle,
  type TemplateCategoryFilter,
} from '@/lib/i18n/strings/search-template-selection';
import type { ROMVersion } from '@/types/rom';

interface TemplateSelectionDialogProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  onApplyTemplate: (seeds: number[]) => void;
}

/** ROMVersion から TemplateVersion へ変換 */
function toTemplateVersion(romVersion: ROMVersion): TemplateVersion {
  return romVersion === 'B' || romVersion === 'W' ? 'BW' : 'BW2';
}

export function TemplateSelectionDialog({
  isOpen,
  onOpenChange,
  onApplyTemplate,
}: TemplateSelectionDialogProps) {
  const [selectedTemplates, setSelectedTemplates] = React.useState<Set<string>>(new Set());
  const [categoryFilter, setCategoryFilter] = React.useState<TemplateCategoryFilter>('all');
  const locale = useLocale();
  const { searchConditions } = useAppStore();
  
  const currentVersion = toTemplateVersion(searchConditions.romVersion);

  // フィルタリングされたテンプレート
  const filteredTemplates = React.useMemo(() => {
    return SEED_TEMPLATES.filter((template) => {
      // バージョンフィルター
      if (template.version !== currentVersion) {
        return false;
      }
      // カテゴリフィルター
      if (categoryFilter !== 'all' && template.category !== categoryFilter) {
        return false;
      }
      return true;
    });
  }, [currentVersion, categoryFilter]);

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

  // ダイアログを開くたびにカテゴリフィルターをリセット
  React.useEffect(() => {
    if (isOpen) {
      setCategoryFilter('all');
      setSelectedTemplates(new Set());
    }
  }, [isOpen]);

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

          {/* Category Filter */}
          <div className="flex items-center gap-2">
            <Label htmlFor="category-filter" className="text-sm font-medium whitespace-nowrap">
              {resolveLocaleValue(templateCategoryFilterLabel, locale)}:
            </Label>
            <Select
              value={categoryFilter}
              onValueChange={(value) => setCategoryFilter(value as TemplateCategoryFilter)}
            >
              <SelectTrigger id="category-filter" className="w-40">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {templateCategoryOptions.map((option) => (
                  <SelectItem key={option} value={option}>
                    {getTemplateCategoryLabel(option, locale)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <ScrollArea className="h-[400px] pr-4">
            <div className="space-y-3">
              {filteredTemplates.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-8">
                  {resolveLocaleValue(templateNoResultsMessage, locale)}
                </p>
              ) : (
                filteredTemplates.map((template: SeedTemplate) => (
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
                ))
              )}
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

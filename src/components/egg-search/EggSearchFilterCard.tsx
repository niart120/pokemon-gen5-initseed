/**
 * EggSearchFilterCard
 * 結果フィルターカード
 */

import React from 'react';
import { Funnel } from '@phosphor-icons/react';
import { PanelCard } from '@/components/ui/panel-card';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { useEggBootTimingSearchStore } from '@/store/egg-boot-timing-search-store';
import { useLocale } from '@/lib/i18n/locale-context';

export function EggSearchFilterCard() {
  const locale = useLocale();
  const { resultFilters, updateResultFilters } = useEggBootTimingSearchStore();

  const labels = {
    title: locale === 'ja' ? 'フィルター' : 'Filter',
    shinyOnly: locale === 'ja' ? '色違いのみ' : 'Shiny Only',
    shinyHint: locale === 'ja' 
      ? '色違いの結果のみ表示' 
      : 'Show only shiny results',
  };

  return (
    <PanelCard
      icon={<Funnel size={20} className="opacity-80" />}
      title={labels.title}
    >
      <div className="space-y-4">
        {/* 色違いフィルター */}
        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <Label htmlFor="shiny-only">{labels.shinyOnly}</Label>
            <p className="text-xs text-muted-foreground">{labels.shinyHint}</p>
          </div>
          <Switch
            id="shiny-only"
            checked={resultFilters.shinyOnly ?? false}
            onCheckedChange={(checked) => updateResultFilters({ shinyOnly: checked })}
          />
        </div>
      </div>
    </PanelCard>
  );
}

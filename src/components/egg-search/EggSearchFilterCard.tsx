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
import {
  eggSearchFilterCardTitle,
  eggSearchFilterLabels,
} from '@/lib/i18n/strings/egg-search';

export function EggSearchFilterCard() {
  const locale = useLocale();
  const { resultFilters, updateResultFilters } = useEggBootTimingSearchStore();

  return (
    <PanelCard
      icon={<Funnel size={20} className="opacity-80" />}
      title={eggSearchFilterCardTitle[locale]}
    >
      <div className="space-y-4">
        {/* 色違いフィルター */}
        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <Label htmlFor="shiny-only">{eggSearchFilterLabels.shinyOnly[locale]}</Label>
            <p className="text-xs text-muted-foreground">{eggSearchFilterLabels.shinyHint[locale]}</p>
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

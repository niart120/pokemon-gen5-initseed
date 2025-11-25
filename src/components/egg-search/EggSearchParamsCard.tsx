/**
 * EggSearchParamsCard
 * 検索条件パラメータ入力カード
 * 
 * NOTE: Timer0/VCountはProfileから自動取得されるため、このカードには含めない
 */

import React from 'react';
import { Sliders } from '@phosphor-icons/react';
import { PanelCard } from '@/components/ui/panel-card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { useEggBootTimingSearchStore } from '@/store/egg-boot-timing-search-store';
import { useLocale } from '@/lib/i18n/locale-context';
import {
  eggSearchParamsCardTitle,
  eggSearchParamsLabels,
} from '@/lib/i18n/strings/egg-search';

// デフォルト値定数
const DEFAULT_FRAME = 8;
const DEFAULT_ADVANCE_COUNT = 1000;

export function EggSearchParamsCard() {
  const locale = useLocale();
  const { draftParams, updateDraftParams, status } = useEggBootTimingSearchStore();
  
  const isRunning = status === 'running' || status === 'starting' || status === 'stopping';

  // 日時をローカルタイムで表示するためのフォーマット
  const formatDatetimeLocal = (isoString: string): string => {
    try {
      const date = new Date(isoString);
      if (isNaN(date.getTime())) return '';
      // datetime-local形式: YYYY-MM-DDTHH:mm:ss
      const pad = (n: number) => n.toString().padStart(2, '0');
      return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
    } catch {
      return '';
    }
  };

  const handleDatetimeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const localValue = e.target.value;
    if (!localValue) return;
    try {
      const date = new Date(localValue);
      updateDraftParams({ startDatetimeIso: date.toISOString() });
    } catch {
      // Invalid date, ignore
    }
  };

  return (
    <PanelCard
      icon={<Sliders size={20} className="opacity-80" />}
      title={eggSearchParamsCardTitle[locale]}
      scrollMode="content"
    >
      <div className="space-y-4">
        {/* 開始日時 */}
        <div className="space-y-2">
          <Label htmlFor="start-datetime">{eggSearchParamsLabels.startDatetime[locale]}</Label>
          <Input
            id="start-datetime"
            type="datetime-local"
            value={formatDatetimeLocal(draftParams.startDatetimeIso)}
            onChange={handleDatetimeChange}
            disabled={isRunning}
            step="1"
          />
        </div>

        {/* 検索範囲（秒） */}
        <div className="space-y-2">
          <Label htmlFor="range-seconds">{eggSearchParamsLabels.rangeSeconds[locale]}</Label>
          <Input
            id="range-seconds"
            type="number"
            min={1}
            max={86400}
            value={draftParams.rangeSeconds}
            onChange={(e) =>
              updateDraftParams({ rangeSeconds: parseInt(e.target.value, 10) || 1 })
            }
            disabled={isRunning}
          />
        </div>

        {/* フレーム */}
        <div className="space-y-2">
          <Label htmlFor="frame">{eggSearchParamsLabels.frame[locale]}</Label>
          <Input
            id="frame"
            type="number"
            min={0}
            max={255}
            value={draftParams.frame}
            onChange={(e) =>
              updateDraftParams({ frame: parseInt(e.target.value, 10) || DEFAULT_FRAME })
            }
            disabled={isRunning}
          />
        </div>

        {/* 開始Advance */}
        <div className="space-y-2">
          <Label htmlFor="user-offset">{eggSearchParamsLabels.userOffset[locale]}</Label>
          <Input
            id="user-offset"
            type="number"
            min={0}
            value={draftParams.userOffset}
            onChange={(e) =>
              updateDraftParams({ userOffset: parseInt(e.target.value, 10) || 0 })
            }
            disabled={isRunning}
          />
        </div>

        {/* 検索Advance数 */}
        <div className="space-y-2">
          <Label htmlFor="advance-count">{eggSearchParamsLabels.advanceCount[locale]}</Label>
          <Input
            id="advance-count"
            type="number"
            min={1}
            max={100000}
            value={draftParams.advanceCount}
            onChange={(e) =>
              updateDraftParams({ advanceCount: parseInt(e.target.value, 10) || DEFAULT_ADVANCE_COUNT })
            }
            disabled={isRunning}
          />
        </div>

        {/* キー入力マスク */}
        <div className="space-y-2">
          <Label htmlFor="key-input-mask">{eggSearchParamsLabels.keyInput[locale]}</Label>
          <Input
            id="key-input-mask"
            type="text"
            placeholder="0x0000"
            value={`0x${draftParams.keyInputMask.toString(16).toUpperCase().padStart(4, '0')}`}
            onChange={(e) => {
              const value = e.target.value.replace(/^0x/i, '');
              const parsed = parseInt(value, 16);
              if (!isNaN(parsed)) {
                updateDraftParams({ keyInputMask: parsed & 0xffff });
              }
            }}
            disabled={isRunning}
            className="font-mono"
          />
        </div>
      </div>
    </PanelCard>
  );
}

import React from 'react';
import { formatElapsedTime, formatRemainingTime, formatProcessingRate } from '@/lib/utils/format-helpers';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import {
  searchProgressTimeElapsedLabel,
  searchProgressTimeRemainingLabel,
  searchProgressTimeSpeedLabel,
} from '@/lib/i18n/strings/search-progress';

interface TimeDisplayProps {
  elapsedTime: number;
  estimatedTimeRemaining: number;
  currentStep: number;
  totalSteps?: number;
  /** 処理速度計算用の実効進捗（指定時はcurrentStepの代わりに使用） */
  effectiveProgress?: number;
  /** 処理速度計算用の処理済み秒数（最優先で使用） */
  processedSeconds?: number;
}

/**
 * 統一時間表示コンポーネント
 * 直列・並列検索の両方で共通の時間情報を表示
 */
export function TimeDisplay({ 
  elapsedTime, 
  estimatedTimeRemaining, 
  currentStep, 
  totalSteps: _,
  effectiveProgress,
  processedSeconds,
}: TimeDisplayProps) {
  // 処理速度計算：processedSeconds > effectiveProgress > currentStep の優先順
  const progressForRate = processedSeconds ?? effectiveProgress ?? currentStep;
  const processingRate = formatProcessingRate(progressForRate, elapsedTime);
  const locale = useLocale();
  const elapsedLabel = resolveLocaleValue(searchProgressTimeElapsedLabel, locale);
  const remainingLabel = resolveLocaleValue(searchProgressTimeRemainingLabel, locale);
  const speedLabel = resolveLocaleValue(searchProgressTimeSpeedLabel, locale);
  
  return (
    <div className="grid grid-cols-3 gap-3 text-xs">
      <div>
        <div className="text-muted-foreground">{elapsedLabel}</div>
        <div className="font-mono text-sm">
          {formatElapsedTime(elapsedTime)}
        </div>
      </div>
      <div>
        <div className="text-muted-foreground">{remainingLabel}</div>
        <div className="font-mono text-sm">
          {formatRemainingTime(estimatedTimeRemaining)}
        </div>
      </div>
      <div>
        <div className="text-muted-foreground">{speedLabel}</div>
        <div className="font-mono text-sm">
          {processingRate}
        </div>
      </div>
    </div>
  );
}

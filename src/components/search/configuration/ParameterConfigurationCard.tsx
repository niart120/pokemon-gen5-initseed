import React from 'react';
import { Card } from '@/components/ui/card';
import { StandardCardHeader, StandardCardContent } from '@/components/ui/card-helpers';
import { Separator } from '@/components/ui/separator';
import { Gear } from '@phosphor-icons/react';
import { Timer0VCountParam } from './params/Timer0VCountParam';
import { DateRangeParam } from './params/DateRangeParam';
import { MACAddressParam } from './params/MACAddressParam';

export function ParameterConfigurationCard() {
  return (
    <Card className="py-2 flex flex-col h-full gap-2">
      <StandardCardHeader icon={<Gear size={20} className="opacity-80" />} title="Parameters" />
      <StandardCardContent>
        {/* Timer0 & VCount 設定 */}
        <Timer0VCountParam />
        
        <Separator />
        
        {/* 日付範囲設定 */}
        <DateRangeParam />
        
        <Separator />
        
        {/* MACアドレス設定 */}
        <MACAddressParam />
      </StandardCardContent>
    </Card>
  );
}

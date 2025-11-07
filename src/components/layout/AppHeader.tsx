import React from 'react';
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from '@/components/ui/select';
import { SUPPORTED_LOCALES, type SupportedLocale } from '@/types/i18n';
import { useLocale, useLocaleControls } from '@/lib/i18n/locale-context';

export function AppHeader() {
  const locale = useLocale();
  const { setLocale } = useLocaleControls();

  const localeLabels: Record<SupportedLocale, string> = {
    ja: '日本語',
    en: 'English',
  };

  const handleLocaleChange = (value: string) => {
    setLocale(value as SupportedLocale);
  };

  return (
    <header className="border-b bg-card">
      <div className="container mx-auto px-4 py-2">
        <div className="flex items-center justify-between gap-2 min-w-0">
          <div className="min-w-0 flex-1">
            <h1 className="text-lg sm:text-xl font-bold text-foreground truncate">
              Pokémon BW/BW2 Initial Seed Search
            </h1>
            <p className="text-muted-foreground text-xs mt-0.5 hidden sm:block">
              Advanced seed calculation for competitive RNG
            </p>
          </div>
          <div className="flex items-center gap-2 sm:gap-3 flex-shrink-0">
            <div className="flex items-center gap-1.5 sm:gap-2">
              <span className="text-xs sm:text-sm text-muted-foreground whitespace-nowrap">Locale</span>
              <Select value={locale} onValueChange={handleLocaleChange}>
                <SelectTrigger className="w-[110px] h-8">
                  <SelectValue>{localeLabels[locale]}</SelectValue>
                </SelectTrigger>
                <SelectContent align="end" className="min-w-[120px]">
                  {SUPPORTED_LOCALES.map(code => (
                    <SelectItem key={code} value={code}>{localeLabels[code]}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

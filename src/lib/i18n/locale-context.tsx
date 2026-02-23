import React, { createContext, useContext, useEffect, useMemo } from 'react';
import { useAppStore } from '@/store/app-store';
import type { SupportedLocale } from '@/types/i18n';

interface LocaleContextValue {
  locale: SupportedLocale;
  setLocale: (next: SupportedLocale) => void;
}

const LocaleContext = createContext<LocaleContextValue | undefined>(undefined);

interface LocaleProviderProps {
  children: React.ReactNode;
}

 
export const LocaleProvider: React.FC<LocaleProviderProps> = ({ children }) => {
  const locale = useAppStore(s => s.locale);
  const setLocale = useAppStore(s => s.setLocale);

  useEffect(() => {
    document.documentElement.lang = locale;
  }, [locale]);

  const value = useMemo<LocaleContextValue>(() => ({ locale, setLocale }), [locale, setLocale]);

  return <LocaleContext.Provider value={value}>{children}</LocaleContext.Provider>;
};

// eslint-disable-next-line react-refresh/only-export-components
export function useLocale(): SupportedLocale {
  const ctx = useContext(LocaleContext);
  const fallbackLocale = useAppStore((state) => state.locale);
  if (!ctx) {
    return fallbackLocale;
  }
  return ctx.locale;
}

// eslint-disable-next-line react-refresh/only-export-components
export function useLocaleControls(): LocaleContextValue {
  const ctx = useContext(LocaleContext);
  const locale = useAppStore((state) => state.locale);
  const setLocale = useAppStore((state) => state.setLocale);
  if (!ctx) {
    return { locale, setLocale };
  }
  return ctx;
}

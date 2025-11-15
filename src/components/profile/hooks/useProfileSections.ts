import React from 'react';
import type { SectionKey, SectionState } from './profileFormTypes';

export function useProfileSections(isStack: boolean): {
  sectionOpen: SectionState;
  toggleSection: (key: SectionKey) => void;
} {
  const sectionDefaults = React.useMemo<SectionState>(
    () =>
      isStack
        ? { profile: true, rom: true, timer: false, game: false }
        : { profile: true, rom: true, timer: true, game: true },
    [isStack],
  );

  const [sectionOpen, setSectionOpen] = React.useState<SectionState>(sectionDefaults);

  React.useEffect(() => {
    setSectionOpen(sectionDefaults);
  }, [sectionDefaults]);

  const toggleSection = React.useCallback((key: SectionKey) => {
    setSectionOpen((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  return { sectionOpen, toggleSection };
}

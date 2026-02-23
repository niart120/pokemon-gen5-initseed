import React from 'react';
import { CaretDown } from '@phosphor-icons/react';
import type { SectionKey } from '../hooks/profileFormTypes';

interface CollapsedSectionProps {
  sectionKey: SectionKey;
  headingId: string;
  title: string;
  isOpen: boolean;
  onToggle: (section: SectionKey) => void;
  children: React.ReactNode;
}

export function CollapsedSection({ sectionKey, headingId, title, isOpen, onToggle, children }: CollapsedSectionProps) {
  return (
    <section
      aria-labelledby={headingId}
      className="rounded-xl border bg-card/60 px-3 py-2"
    >
      <button
        type="button"
        className="flex w-full items-center justify-between gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground"
        aria-expanded={isOpen}
        aria-controls={`${headingId}-content`}
        onClick={() => onToggle(sectionKey)}
      >
        <span id={headingId}>{title}</span>
        <CaretDown size={16} className={`transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>
      <div
        id={`${headingId}-content`}
        className={isOpen ? 'mt-3 space-y-3' : 'hidden'}
      >
        {children}
      </div>
    </section>
  );
}

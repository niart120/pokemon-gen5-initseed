import type { Hardware, ROMRegion, ROMVersion } from '@/types/rom';

export type SectionKey = 'profile' | 'rom' | 'timer' | 'game';
export type SectionState = Record<SectionKey, boolean>;

export interface ProfileFormState {
  name: string;
  description: string;
  romVersion: ROMVersion;
  romRegion: ROMRegion;
  hardware: Hardware;
  timer0Auto: boolean;
  timer0Min: string;
  timer0Max: string;
  vcountMin: string;
  vcountMax: string;
  frame: number;
  macSegments: string[];
  tid: string;
  sid: string;
  shinyCharm: boolean;
  newGame: boolean;
  withSave: boolean;
  memoryLink: boolean;
}

import { create } from 'zustand';
import type { DeviceProfileDraft } from '@/types/profile';

interface ProfileFormStore {
  isDirty: boolean;
  draft: DeviceProfileDraft | null;
  validationErrors: string[];
  setDirty: (dirty: boolean) => void;
  setDraft: (draft: DeviceProfileDraft | null, validationErrors?: string[]) => void;
  reset: () => void;
}

export const useProfileFormStore = create<ProfileFormStore>((set) => ({
  isDirty: false,
  draft: null,
  validationErrors: [],
  setDirty: (dirty) => set({ isDirty: dirty }),
  setDraft: (draft, validationErrors = []) => set({ draft, validationErrors }),
  reset: () => set({ isDirty: false, draft: null, validationErrors: [] }),
}));

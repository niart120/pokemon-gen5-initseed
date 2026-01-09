/**
 * report-needle-search-store.ts
 * レポ針検索カード用のZustandストア
 */

import { create } from 'zustand';
import type { DeviceProfile, NumericRange } from '@/types/profile';
import type {
  ReportNeedleSearchDraft,
  ReportNeedleSearchMode,
  ReportNeedleSearchResult,
  ReportNeedleSearchStatus,
  ReportNeedleWorkerRequest,
  ReportNeedleWorkerResponse,
  ReportNeedleSearchParams,
  ReportNeedleProfileParams,
} from '@/types/report-needle-search';
import { KEY_INPUT_DEFAULT, normalizeKeyMask } from '@/lib/utils/key-input';

const MAX_NEEDLE_LENGTH = 128;
const DEFAULT_TIME_VALUE = '00:00:00';

function toLocalDateTimeParts(iso?: string): { dateValue: string; timeValue: string } {
  if (!iso) return { dateValue: '', timeValue: '' };
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return { dateValue: '', timeValue: '' };
  const pad = (value: number) => value.toString().padStart(2, '0');
  const year = date.getFullYear();
  const month = pad(date.getMonth() + 1);
  const day = pad(date.getDate());
  const hours = pad(date.getHours());
  const minutes = pad(date.getMinutes());
  const seconds = pad(date.getSeconds());
  return {
    dateValue: `${year}-${month}-${day}`,
    timeValue: `${hours}:${minutes}:${seconds}`,
  };
}

function normalizeTimeValue(value: string): string | null {
  if (!value) return null;
  if (/^\d{2}:\d{2}:\d{2}$/.test(value)) return value;
  if (/^\d{2}:\d{2}$/.test(value)) return `${value}:00`;
  return null;
}

function toIsoStringFromLocal(dateValue: string, timeValue: string): string {
  const normalizedTime = normalizeTimeValue(timeValue);
  if (!dateValue || !normalizedTime) return '';
  const combined = `${dateValue}T${normalizedTime}`;
  const date = new Date(combined);
  if (Number.isNaN(date.getTime())) {
    return '';
  }
  return date.toISOString();
}

function sanitizeNeedle(value: string): string {
  return value.replace(/[^0-7]/g, '').slice(0, MAX_NEEDLE_LENGTH);
}

function normalizeRange(range: NumericRange, bounds: { min: number; max: number }): NumericRange {
  const min = Number.isFinite(range.min) ? Math.max(bounds.min, Math.min(bounds.max, Math.round(range.min))) : bounds.min;
  const max = Number.isFinite(range.max) ? Math.max(bounds.min, Math.min(bounds.max, Math.round(range.max))) : bounds.max;
  if (min > max) {
    return { min: max, max: min };
  }
  return { min, max };
}

function createDefaultDraft(): ReportNeedleSearchDraft {
  return {
    mode: 'initial-seed',
    needleValue: '',
    timer0Range: { min: 3193, max: 3194 },
    vcountRange: { min: 96, max: 96 },
    advanceRange: { start: 0, end: 2000 },
    startup: {
      timestampIso: '',
      keyMask: KEY_INPUT_DEFAULT,
    },
    initialSeed: {
      seedHex: '',
    },
  };
}

// Worker管理用
let worker: Worker | null = null;
let currentProfile: DeviceProfile | null = null;

function createWorker(): Worker {
  return new Worker(
    new URL('../workers/report-needle-search-worker.ts', import.meta.url),
    { type: 'module' }
  );
}

function terminateWorker(): void {
  if (worker) {
    worker.terminate();
    worker = null;
  }
}

interface ReportNeedleSearchState {
  draft: ReportNeedleSearchDraft;
  validationErrors: string[];
  status: ReportNeedleSearchStatus;
  results: ReportNeedleSearchResult[];
  errorMessage: string | null;
}

interface ReportNeedleSearchActions {
  setMode: (mode: ReportNeedleSearchMode) => void;
  updateNeedleValue: (value: string) => void;
  appendNeedleDigit: (digit: string) => void;
  updateRange: (key: 'timer0Range' | 'vcountRange', edge: 'min' | 'max', value: number) => void;
  updateAdvanceRange: (edge: 'start' | 'end', value: number) => void;
  setStartupDate: (value: string) => void;
  setStartupTime: (value: string) => void;
  setStartupKeyMask: (value: number) => void;
  setInitialSeedHex: (value: string) => void;
  applyProfileRanges: (profile: DeviceProfile) => void;
  setCurrentProfile: (profile: DeviceProfile) => void;
  validateDraft: () => boolean;
  startSearch: () => Promise<void>;
  pauseSearch: () => void;
  resumeSearch: () => void;
  stopSearch: () => void;
  reset: () => void;
}

export interface ReportNeedleSearchStore extends ReportNeedleSearchState, ReportNeedleSearchActions {}

export const useReportNeedleSearchStore = create<ReportNeedleSearchStore>((set, get) => ({
  draft: createDefaultDraft(),
  validationErrors: [],
  status: 'idle',
  results: [],
  errorMessage: null,

  setMode: (mode) => {
    set((state) => ({
      draft: { ...state.draft, mode },
      validationErrors: [],
    }));
  },

  updateNeedleValue: (value) => {
    set((state) => ({
      draft: {
        ...state.draft,
        needleValue: sanitizeNeedle(value),
      },
      validationErrors: [],
    }));
  },

  appendNeedleDigit: (digit) => {
    if (!/^[0-7]$/.test(digit)) return;
    set((state) => ({
      draft: {
        ...state.draft,
        needleValue: sanitizeNeedle(`${state.draft.needleValue}${digit}`),
      },
      validationErrors: [],
    }));
  },

  updateRange: (key, edge, value) => {
    set((state) => {
      const nextRange = normalizeRange({ ...state.draft[key], [edge]: value }, { min: 0, max: 65535 });
      return {
        draft: { ...state.draft, [key]: nextRange },
        validationErrors: [],
      };
    });
  },

  updateAdvanceRange: (edge, value) => {
    const clamped = Math.max(0, Math.round(value));
    set((state) => ({
      draft: {
        ...state.draft,
        advanceRange: {
          ...state.draft.advanceRange,
          [edge]: clamped,
        },
      },
      validationErrors: [],
    }));
  },

  setStartupDate: (value) => {
    set((state) => {
      const { timeValue } = toLocalDateTimeParts(state.draft.startup.timestampIso);
      const nextIso = toIsoStringFromLocal(value, timeValue || DEFAULT_TIME_VALUE);
      return {
        draft: {
          ...state.draft,
          startup: {
            ...state.draft.startup,
            timestampIso: nextIso,
          },
        },
        validationErrors: [],
      };
    });
  },

  setStartupTime: (value) => {
    set((state) => {
      const { dateValue } = toLocalDateTimeParts(state.draft.startup.timestampIso);
      const nextIso = dateValue ? toIsoStringFromLocal(dateValue, value) : state.draft.startup.timestampIso;
      return {
        draft: {
          ...state.draft,
          startup: {
            ...state.draft.startup,
            timestampIso: nextIso,
          },
        },
        validationErrors: [],
      };
    });
  },

  setStartupKeyMask: (value) => {
    const masked = normalizeKeyMask(value);
    set((state) => ({
      draft: {
        ...state.draft,
        startup: {
          ...state.draft.startup,
          keyMask: masked,
        },
      },
      validationErrors: [],
    }));
  },

  setInitialSeedHex: (value) => {
    const sanitized = value.replace(/[^0-9a-fA-F]/g, '').slice(0, 16).toUpperCase();
    set((state) => ({
      draft: {
        ...state.draft,
        initialSeed: { ...state.draft.initialSeed, seedHex: sanitized },
      },
      validationErrors: [],
    }));
  },

  applyProfileRanges: (profile) => {
    currentProfile = profile;
    set((state) => ({
      draft: {
        ...state.draft,
        timer0Range: normalizeRange(profile.timer0Range, { min: 0, max: 65535 }),
        vcountRange: normalizeRange(profile.vcountRange, { min: 0, max: 255 }),
      },
      validationErrors: [],
    }));
  },

  setCurrentProfile: (profile) => {
    currentProfile = profile;
  },

  validateDraft: () => {
    const { draft } = get();
    const errors: string[] = [];

    const needle = sanitizeNeedle(draft.needleValue);

    if (needle.length === 0) {
      errors.push('針列が空です。');
    }

    if (needle.length > MAX_NEEDLE_LENGTH) {
      errors.push(`針列は最大 ${MAX_NEEDLE_LENGTH} 桁までです。`);
    }

    const timer0Range = draft.timer0Range;
    if (timer0Range.min > timer0Range.max) {
      errors.push('Timer0範囲が不正です。');
    }

    const vcountRange = draft.vcountRange;
    if (vcountRange.min > vcountRange.max) {
      errors.push('VCount範囲が不正です。');
    }

    if (draft.advanceRange.start > draft.advanceRange.end) {
      errors.push('消費探索範囲の開始/終了を確認してください。');
    }

    if (draft.mode === 'startup') {
      if (!draft.startup.timestampIso) {
        errors.push('起動日時を入力してください。');
      } else {
        const ts = new Date(draft.startup.timestampIso).getTime();
        if (Number.isNaN(ts)) {
          errors.push('起動日時の形式を確認してください。');
        }
      }
    }

    if (draft.mode === 'initial-seed') {
      const seed = draft.initialSeed.seedHex.trim();
      if (!/^[0-9A-Fa-f]+$/.test(seed)) {
        errors.push('初期Seedは16進数で入力してください。');
      }
    }

    set((state) => ({
      validationErrors: errors,
      draft: {
        ...state.draft,
        needleValue: needle,
      },
    }));

    return errors.length === 0;
  },

  startSearch: async () => {
    const isValid = get().validateDraft();
    if (!isValid) return;

    set({ status: 'starting', results: [], errorMessage: null });

    // 既存のWorkerを終了
    terminateWorker();

    // 新しいWorkerを作成
    worker = createWorker();

    const { draft } = get();

    // Workerメッセージハンドラ
    worker.onmessage = (ev: MessageEvent<ReportNeedleWorkerResponse>) => {
      const msg = ev.data;
      switch (msg.type) {
        case 'READY':
          // Workerが準備完了したら検索開始
          if (worker) {
            const params: ReportNeedleSearchParams = {
              mode: draft.mode,
              needleValue: draft.needleValue,
              timer0Range: draft.timer0Range,
              vcountRange: draft.vcountRange,
              advanceRange: draft.advanceRange,
              seedHex: draft.mode === 'initial-seed' ? draft.initialSeed.seedHex : undefined,
              timestampIso: draft.mode === 'startup' ? draft.startup.timestampIso : undefined,
              keyMask: draft.mode === 'startup' ? draft.startup.keyMask : undefined,
              profile: draft.mode === 'startup' && currentProfile
                ? {
                    romVersion: currentProfile.romVersion,
                    romRegion: currentProfile.romRegion,
                    hardware: currentProfile.hardware,
                    macAddress: [...currentProfile.macAddress],
                  } as ReportNeedleProfileParams
                : undefined,
            };
            const request: ReportNeedleWorkerRequest = { type: 'START', params };
            worker.postMessage(request);
            set({ status: 'running' });
          }
          break;
        case 'RESULTS':
          set((state) => ({
            results: [...state.results, ...msg.results],
          }));
          break;
        case 'COMPLETE':
          set({ status: 'completed' });
          terminateWorker();
          break;
        case 'ERROR':
          set({ status: 'error', errorMessage: msg.message });
          terminateWorker();
          break;
        case 'PAUSED':
          set({ status: 'paused' });
          break;
        case 'RESUMED':
          set({ status: 'running' });
          break;
        default:
          break;
      }
    };

    worker.onerror = (ev) => {
      set({ status: 'error', errorMessage: ev.message || 'Worker error' });
      terminateWorker();
    };
  },

  pauseSearch: () => {
    const status = get().status;
    if (status !== 'running') return;
    if (worker) {
      const request: ReportNeedleWorkerRequest = { type: 'PAUSE' };
      worker.postMessage(request);
    }
  },

  resumeSearch: () => {
    const status = get().status;
    if (status !== 'paused') return;
    if (worker) {
      const request: ReportNeedleWorkerRequest = { type: 'RESUME' };
      worker.postMessage(request);
    }
  },

  stopSearch: () => {
    const status = get().status;
    if (status !== 'running' && status !== 'paused') return;
    set({ status: 'stopping' });
    if (worker) {
      const request: ReportNeedleWorkerRequest = { type: 'STOP' };
      worker.postMessage(request);
    }
    terminateWorker();
    set({ status: 'idle' });
  },

  reset: () => {
    terminateWorker();
    set((state) => ({
      draft: {
        ...createDefaultDraft(),
        timer0Range: state.draft.timer0Range,
        vcountRange: state.draft.vcountRange,
      },
      validationErrors: [],
      status: 'idle',
      results: [],
      errorMessage: null,
    }));
  },
}));

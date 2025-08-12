import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { GenerationControlCard } from '@/components/generation/GenerationControlCard';

// useAppStore を丸ごとモック（必要最小限のshapeのみ）
vi.mock('@/store/app-store', () => ({
  useAppStore: vi.fn(),
}));
import { useAppStore } from '@/store/app-store';

interface MockState {
  status: string;
  validationErrors: string[];
  lastCompletion?: { reason: string } | null;
  validateDraft: () => void;
  startGeneration: () => Promise<void>;
  pauseGeneration: () => void;
  resumeGeneration: () => void;
  stopGeneration: () => void;
}

function setupMock(partial: Partial<MockState>) {
  const base: MockState = {
    status: 'idle',
    validationErrors: [],
    lastCompletion: null,
    validateDraft: vi.fn(),
    startGeneration: vi.fn().mockResolvedValue(undefined),
    pauseGeneration: vi.fn(),
    resumeGeneration: vi.fn(),
    stopGeneration: vi.fn(),
  };
  (useAppStore as unknown as { mockImplementation: (impl: any) => void }).mockImplementation((sel?: any) => {
    const state = { ...base, ...partial } as MockState;
    return sel ? sel(state) : state;
  });
}

describe('GenerationControlCard UI states', () => {
  it('shows Start button in idle', () => {
    setupMock({ status: 'idle' });
    render(<GenerationControlCard />);
    expect(screen.getByTestId('gen-start-btn')).toBeInTheDocument();
    expect(screen.queryByTestId('gen-pause-btn')).toBeNull();
  });
  it('shows Pause + Stop in running', () => {
    setupMock({ status: 'running' });
    render(<GenerationControlCard />);
    expect(screen.getByTestId('gen-pause-btn')).toBeInTheDocument();
    expect(screen.getByTestId('gen-stop-btn')).toBeInTheDocument();
    expect(screen.queryByTestId('gen-start-btn')).toBeNull();
  });
  it('shows Resume + Stop in paused', () => {
    setupMock({ status: 'paused' });
    render(<GenerationControlCard />);
    expect(screen.getByTestId('gen-resume-btn')).toBeInTheDocument();
    expect(screen.getByTestId('gen-stop-btn')).toBeInTheDocument();
  });
  it('shows Start again in completed', () => {
    setupMock({ status: 'completed', lastCompletion: { reason: 'max-advances' } });
    render(<GenerationControlCard />);
    expect(screen.getByTestId('gen-start-btn')).toBeInTheDocument();
  });
  it('shows validation errors', () => {
    setupMock({ status: 'idle', validationErrors: ['Err1','Err2'] });
    render(<GenerationControlCard />);
    expect(screen.getByTestId('gen-validation-errors').textContent).toContain('Err1');
  });
  it('calls correct actions across transitions', async () => {
    const start = vi.fn().mockResolvedValue(undefined);
    const pause = vi.fn();
    const resume = vi.fn();
    const stop = vi.fn();
    // idle -> running
    setupMock({ status: 'idle', startGeneration: start });
    render(<GenerationControlCard />);
    fireEvent.click(screen.getByTestId('gen-start-btn'));
    expect(start).toHaveBeenCalledTimes(1);
    // running -> paused
    setupMock({ status: 'running', pauseGeneration: pause });
    render(<GenerationControlCard />);
    fireEvent.click(screen.getByTestId('gen-pause-btn'));
    expect(pause).toHaveBeenCalledTimes(1);
    // paused -> running (resume) then stop
    setupMock({ status: 'paused', resumeGeneration: resume, stopGeneration: stop });
    render(<GenerationControlCard />);
    fireEvent.click(screen.getByTestId('gen-resume-btn'));
    expect(resume).toHaveBeenCalledTimes(1);
  const stopButtons = screen.getAllByTestId('gen-stop-btn');
  fireEvent.click(stopButtons[stopButtons.length - 1]);
    expect(stop).toHaveBeenCalledTimes(1);
  });
});

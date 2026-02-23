import React from 'react';

interface UseMacAddressInputParams {
  macSegments: string[];
  updateSegments: (updater: (prev: string[]) => string[]) => void;
}

export function useMacAddressInput({ macSegments, updateSegments }: UseMacAddressInputParams) {
  const macInputRefs = React.useRef<Array<HTMLInputElement | null>>([]);

  const focusMacSegment = React.useCallback((index: number) => {
    const field = macInputRefs.current[index];
    if (field) {
      field.focus();
      field.select();
    }
  }, []);

  const handleMacSegmentChange = React.useCallback(
    (index: number, rawValue: string) => {
      const sanitized = rawValue.replace(/[^0-9a-fA-F]/g, '').toUpperCase().slice(0, 2);
      const currentValue = macSegments[index];
      if (sanitized !== currentValue) {
        updateSegments((prev) => {
          const next = [...prev];
          next[index] = sanitized;
          return next;
        });
      }
      if (sanitized.length === 2 && currentValue.length < 2 && index < macSegments.length - 1) {
        focusMacSegment(index + 1);
      }
    },
    [focusMacSegment, macSegments, updateSegments],
  );

  const handleMacSegmentFocus = React.useCallback((event: React.FocusEvent<HTMLInputElement>) => {
    event.target.select();
  }, []);

  const handleMacSegmentMouseDown = React.useCallback((event: React.MouseEvent<HTMLInputElement>) => {
    if (event.currentTarget === document.activeElement) {
      event.preventDefault();
      event.currentTarget.select();
    }
  }, []);

  const handleMacSegmentClick = React.useCallback((event: React.MouseEvent<HTMLInputElement>) => {
    event.currentTarget.select();
  }, []);

  const handleMacSegmentKeyDown = React.useCallback(
    (index: number, event: React.KeyboardEvent<HTMLInputElement>) => {
      const input = event.currentTarget;
      const selectionStart = input.selectionStart ?? 0;
      const selectionEnd = input.selectionEnd ?? 0;
      if (event.key === 'ArrowLeft' && selectionStart === 0 && selectionEnd === 0 && index > 0) {
        event.preventDefault();
        focusMacSegment(index - 1);
      }
      if (
        event.key === 'ArrowRight' &&
        selectionStart === input.value.length &&
        selectionEnd === input.value.length &&
        index < macSegments.length - 1
      ) {
        event.preventDefault();
        focusMacSegment(index + 1);
      }
    },
    [focusMacSegment, macSegments.length],
  );

  const handleMacSegmentPaste = React.useCallback(
    (index: number, event: React.ClipboardEvent<HTMLInputElement>) => {
      const pasted = event.clipboardData.getData('text');
      const sanitized = pasted.replace(/[^0-9a-fA-F]/g, '').toUpperCase();
      if (!sanitized) {
        return;
      }
      event.preventDefault();
      const segmentCapacity = (macSegments.length - index) * 2;
      const usableDigits = Math.min(sanitized.length, segmentCapacity);
      if (usableDigits === 0) {
        return;
      }
      let changed = false;
      updateSegments((prev) => {
        const next = [...prev];
        let cursor = 0;
        for (let i = index; i < next.length && cursor < usableDigits; i += 1) {
          const segmentValue = sanitized.slice(cursor, cursor + 2);
          if (segmentValue !== next[i]) {
            next[i] = segmentValue;
            changed = true;
          }
          cursor += 2;
        }
        return next;
      });
      if (changed) {
        const segmentsAdvanced = Math.floor(usableDigits / 2);
        const hasRemainder = usableDigits % 2 !== 0;
        let targetIndex = index + segmentsAdvanced;
        if (!hasRemainder && usableDigits > 0 && targetIndex < macSegments.length - 1) {
          targetIndex += 1;
        }
        if (targetIndex >= macSegments.length) {
          targetIndex = macSegments.length - 1;
        }
        focusMacSegment(targetIndex);
      }
    },
    [focusMacSegment, macSegments.length, updateSegments],
  );

  return {
    macInputRefs,
    handleMacSegmentChange,
    handleMacSegmentFocus,
    handleMacSegmentMouseDown,
    handleMacSegmentClick,
    handleMacSegmentKeyDown,
    handleMacSegmentPaste,
  };
}

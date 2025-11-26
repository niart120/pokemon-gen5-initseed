/**
 * Input要素のフォーカス時の全選択ハンドラ
 * 
 * date/datetime-local タイプを除くinput要素でフォーカス時に内容を全選択する。
 */

import type { FocusEvent } from 'react';

/**
 * input要素のフォーカス時に内容を全選択する
 * date/datetime-local タイプは除外
 */
export function selectAllOnFocus(e: FocusEvent<HTMLInputElement>): void {
  const input = e.target;
  const type = input.type;
  
  // date/datetime系は除外
  if (type === 'date' || type === 'datetime-local' || type === 'time' || type === 'month' || type === 'week') {
    return;
  }
  
  // 内容を全選択
  input.select();
}

/**
 * カスタムonFocus処理と全選択を組み合わせるヘルパー
 */
export function createFocusHandler(
  customOnFocus?: (e: FocusEvent<HTMLInputElement>) => void
): (e: FocusEvent<HTMLInputElement>) => void {
  return (e: FocusEvent<HTMLInputElement>) => {
    selectAllOnFocus(e);
    customOnFocus?.(e);
  };
}

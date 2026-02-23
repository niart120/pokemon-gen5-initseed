/**
 * KeyInputDialog
 * 
 * キー入力設定用の統一ダイアログコンポーネント。
 * SearchParamsCard / EggSearchParamsCard / BootTimingControls / EggBootTimingControls で共通利用。
 */

import React from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Toggle } from '@/components/ui/toggle';
import type { KeyName } from '@/lib/utils/key-input';

// レイアウト定数
const SHOULDER_KEYS: KeyName[] = ['L', 'R'];
const START_SELECT_KEYS: KeyName[] = ['Select', 'Start'];
const DPAD_LAYOUT: Array<Array<KeyName | null>> = [
  [null, '[↑]', null],
  ['[←]', null, '[→]'],
  [null, '[↓]', null],
];
const FACE_BUTTON_LAYOUT: Array<Array<KeyName | null>> = [
  [null, 'X', null],
  ['Y', null, 'A'],
  [null, 'B', null],
];

// アクセシビリティラベル
const KEY_ACCESSIBILITY_LABELS: Partial<Record<KeyName, string>> = {
  '[↑]': 'Up',
  '[↓]': 'Down',
  '[←]': 'Left',
  '[→]': 'Right',
};

export interface KeyInputDialogLabels {
  dialogTitle: string;
  reset: string;
  apply: string;
}

export interface KeyInputDialogProps {
  /** ダイアログの開閉状態 */
  isOpen: boolean;
  /** 開閉状態の変更ハンドラ */
  onOpenChange: (open: boolean) => void;
  /** 現在選択されているキー名の配列 */
  availableKeys: KeyName[];
  /** キートグル時のハンドラ */
  onToggleKey: (key: KeyName) => void;
  /** リセットボタン押下時のハンドラ */
  onReset: () => void;
  /** 適用ボタン押下時のハンドラ */
  onApply: () => void;
  /** ラベル（i18n対応） */
  labels: KeyInputDialogLabels;
  /** ダイアログの最大幅クラス（デフォルト: sm:max-w-md） */
  maxWidthClass?: string;
}

/**
 * キートグルを描画するヘルパー
 */
function KeyToggle({
  keyName,
  pressed,
  onToggle,
  className,
}: {
  keyName: KeyName;
  pressed: boolean;
  onToggle: () => void;
  className?: string;
}) {
  // 方向キーの表示テキストから括弧を除去
  const displayText = keyName.startsWith('[') && keyName.endsWith(']')
    ? keyName.slice(1, -1)
    : keyName;
  const ariaLabel = KEY_ACCESSIBILITY_LABELS[keyName] ?? keyName;

  return (
    <Toggle
      value={keyName}
      aria-label={ariaLabel}
      pressed={pressed}
      onPressedChange={onToggle}
      className={className}
    >
      {displayText}
    </Toggle>
  );
}

/**
 * 共通キー入力設定ダイアログ
 */
export const KeyInputDialog: React.FC<KeyInputDialogProps> = ({
  isOpen,
  onOpenChange,
  availableKeys,
  onToggleKey,
  onReset,
  onApply,
  labels,
  maxWidthClass = 'sm:max-w-md',
}) => {
  const isPressed = (key: KeyName) => availableKeys.includes(key);

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className={maxWidthClass}>
        <DialogHeader>
          <DialogTitle>{labels.dialogTitle}</DialogTitle>
        </DialogHeader>
        <div className="space-y-6 py-4">
          {/* Shoulder Keys (L / R) */}
          <div className="flex justify-between px-8">
            {SHOULDER_KEYS.map((key) => (
              <KeyToggle
                key={key}
                keyName={key}
                pressed={isPressed(key)}
                onToggle={() => onToggleKey(key)}
                className="px-6 py-2"
              />
            ))}
          </div>

          {/* Main Grid: D-Pad / Start+Select / Face Buttons */}
          <div className="grid grid-cols-3 gap-4">
            {/* D-Pad */}
            <div className="flex flex-col items-center justify-center space-y-2">
              <div className="grid grid-cols-3 gap-1 font-arrows">
                {DPAD_LAYOUT.flat().map((key, index) =>
                  key ? (
                    <KeyToggle
                      key={`dpad-${key}`}
                      keyName={key}
                      pressed={isPressed(key)}
                      onToggle={() => onToggleKey(key)}
                      className="w-12 h-12"
                    />
                  ) : (
                    <span key={`dpad-blank-${index}`} className="w-12 h-12" />
                  )
                )}
              </div>
            </div>

            {/* Start / Select */}
            <div className="flex flex-col items-center justify-center space-y-2">
              <div className="flex gap-2">
                {START_SELECT_KEYS.map((key) => (
                  <KeyToggle
                    key={key}
                    keyName={key}
                    pressed={isPressed(key)}
                    onToggle={() => onToggleKey(key)}
                    className="px-3 py-2"
                  />
                ))}
              </div>
            </div>

            {/* Face Buttons (X, Y, A, B) */}
            <div className="flex flex-col items-center justify-center space-y-2">
              <div className="grid grid-cols-3 gap-1">
                {FACE_BUTTON_LAYOUT.flat().map((key, index) =>
                  key ? (
                    <KeyToggle
                      key={`face-${key}`}
                      keyName={key}
                      pressed={isPressed(key)}
                      onToggle={() => onToggleKey(key)}
                      className="w-12 h-12"
                    />
                  ) : (
                    <span key={`face-blank-${index}`} className="w-12 h-12" />
                  )
                )}
              </div>
            </div>
          </div>

          {/* Footer: Reset / Apply */}
          <div className="flex justify-between items-center pt-4 border-t">
            <Button type="button" variant="outline" size="sm" onClick={onReset}>
              {labels.reset}
            </Button>
            <Button type="button" size="sm" onClick={onApply}>
              {labels.apply}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { Toggle } from '@/components/ui/toggle';
import { useAppStore } from '../../../../store/app-store';
import { GameController } from '@phosphor-icons/react';

// キーマッピング定義（下位ビットから）
const KEY_BITS = {
  A: 0,
  B: 1,
  Select: 2,
  Start: 3,
  Right: 4,
  Left: 5,
  Up: 6,
  Down: 7,
  R: 8,
  L: 9,
  X: 10,
  Y: 11,
} as const;

// キーの種類
type KeyName = keyof typeof KEY_BITS;

// デフォルト値（0 = キー入力なし）
const DEFAULT_KEY_INPUT = 0x0000;

// keyInput（mask）から有効なキー配列を生成
function getAvailableKeys(keyInput: number): KeyName[] {
  const available: KeyName[] = [];
  
  for (const [key, bit] of Object.entries(KEY_BITS)) {
    // ビットが1の場合、利用可能
    if ((keyInput & (1 << bit)) !== 0) {
      available.push(key as KeyName);
    }
  }
  return available;
}

// 有効なキー配列からkeyInput（mask）を生成
function calculateKeyInput(availableKeys: KeyName[]): number {
  let mask = 0;
  
  for (const key of availableKeys) {
    const bit = KEY_BITS[key];
    // 利用可能なキーのビットを1にする
    mask |= (1 << bit);
  }
  
  return mask;
}

export function KeyInputParam() {
  const { searchConditions, setSearchConditions } = useAppStore();
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [tempKeyInput, setTempKeyInput] = useState(searchConditions.keyInput);
  
  // 現在利用可能なキーのリスト
  const availableKeys = getAvailableKeys(searchConditions.keyInput);
  const tempAvailableKeys = getAvailableKeys(tempKeyInput);
  
  const handleToggleKey = (key: KeyName) => {
    const currentAvailable = getAvailableKeys(tempKeyInput);
    let newAvailable: KeyName[];
    
    if (currentAvailable.includes(key)) {
      // キーを外す
      newAvailable = currentAvailable.filter(k => k !== key);
    } else {
      // キーを追加
      newAvailable = [...currentAvailable, key];
    }
    
    const newKeyInput = calculateKeyInput(newAvailable);
    setTempKeyInput(newKeyInput);
  };

  const handleReset = () => {
    setTempKeyInput(DEFAULT_KEY_INPUT);
  };
  
  const handleApply = () => {
    setSearchConditions({ keyInput: tempKeyInput });
    setIsDialogOpen(false);
  };
  
  const handleOpenDialog = () => {
    setTempKeyInput(searchConditions.keyInput);
    setIsDialogOpen(true);
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-sm font-medium">Key Input</div>
        <Button
          variant="outline"
          size="sm"
          onClick={handleOpenDialog}
          className="gap-2"
        >
          <GameController size={16} />
          Configure
        </Button>
      </div>
      
      {availableKeys.length > 0 && (
        <div className="text-xs text-muted-foreground">
          {availableKeys.join(', ')}
        </div>
      )}
      
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Key Input Configuration</DialogTitle>
          </DialogHeader>
          
          <div className="space-y-6 py-4">
            {/* L/Rボタン (最上部) */}
            <div className="flex justify-between px-8">
              <Toggle
                value="L"
                aria-label="L"
                pressed={tempAvailableKeys.includes('L')}
                onPressedChange={() => handleToggleKey('L')}
                className="px-6 py-2"
              >
                L
              </Toggle>
              <Toggle
                value="R"
                aria-label="R"
                pressed={tempAvailableKeys.includes('R')}
                onPressedChange={() => handleToggleKey('R')}
                className="px-6 py-2"
              >
                R
              </Toggle>
            </div>
            
            {/* コントローラレイアウト */}
            <div className="grid grid-cols-3 gap-4">
              {/* 左エリア: 十字キー */}
              <div className="flex flex-col items-center justify-center space-y-2">
                <div className="grid grid-cols-3 gap-1">
                  <div></div>
                  <Toggle
                    value="Up"
                    aria-label="Up"
                    pressed={tempAvailableKeys.includes('Up')}
                    onPressedChange={() => handleToggleKey('Up')}
                    className="w-12 h-12"
                  >
                    Up
                  </Toggle>
                  <div></div>
                  
                  <Toggle
                    value="Left"
                    aria-label="Left"
                    pressed={tempAvailableKeys.includes('Left')}
                    onPressedChange={() => handleToggleKey('Left')}
                    className="w-12 h-12"
                  >
                    Left
                  </Toggle>
                  <div className="w-12 h-12"></div>
                  <Toggle
                    value="Right"
                    aria-label="Right"
                    pressed={tempAvailableKeys.includes('Right')}
                    onPressedChange={() => handleToggleKey('Right')}
                    className="w-12 h-12"
                  >
                    Right
                  </Toggle>
                  
                  <div></div>
                  <Toggle
                    value="Down"
                    aria-label="Down"
                    pressed={tempAvailableKeys.includes('Down')}
                    onPressedChange={() => handleToggleKey('Down')}
                    className="w-12 h-12"
                  >
                    Down
                  </Toggle>
                  <div></div>
                </div>
              </div>
              
              {/* 中央エリア: Select/Start */}
              <div className="flex flex-col items-center justify-center space-y-2">
                <div className="flex gap-2">
                  <Toggle
                    value="Select"
                    aria-label="Select"
                    pressed={tempAvailableKeys.includes('Select')}
                    onPressedChange={() => handleToggleKey('Select')}
                    className="px-3 py-2"
                  >
                    Select
                  </Toggle>
                  <Toggle
                    value="Start"
                    aria-label="Start"
                    pressed={tempAvailableKeys.includes('Start')}
                    onPressedChange={() => handleToggleKey('Start')}
                    className="px-3 py-2"
                  >
                    Start
                  </Toggle>
                </div>
              </div>
              
              {/* 右エリア: A/B/X/Y */}
              <div className="flex flex-col items-center justify-center space-y-2">
                <div className="grid grid-cols-3 gap-1">
                  <div></div>
                  <Toggle
                    value="X"
                    aria-label="X"
                    pressed={tempAvailableKeys.includes('X')}
                    onPressedChange={() => handleToggleKey('X')}
                    className="w-12 h-12"
                  >
                    X
                  </Toggle>
                  <div></div>
                  
                  <Toggle
                    value="Y"
                    aria-label="Y"
                    pressed={tempAvailableKeys.includes('Y')}
                    onPressedChange={() => handleToggleKey('Y')}
                    className="w-12 h-12"
                  >
                    Y
                  </Toggle>
                  <div className="w-12 h-12"></div>
                  <Toggle
                    value="A"
                    aria-label="A"
                    pressed={tempAvailableKeys.includes('A')}
                    onPressedChange={() => handleToggleKey('A')}
                    className="w-12 h-12"
                  >
                    A
                  </Toggle>
                  
                  <div></div>
                  <Toggle
                    value="B"
                    aria-label="B"
                    pressed={tempAvailableKeys.includes('B')}
                    onPressedChange={() => handleToggleKey('B')}
                    className="w-12 h-12"
                  >
                    B
                  </Toggle>
                  <div></div>
                </div>
              </div>
            </div>
            
            {/* ボタン群 */}
            <div className="flex justify-between items-center pt-4 border-t">
              <Button
                variant="outline"
                size="sm"
                onClick={handleReset}
              >
                Reset All
              </Button>
              <Button
                size="sm"
                onClick={handleApply}
              >
                Apply
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

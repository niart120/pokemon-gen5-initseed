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

// デフォルト値（全ビット1 = 押されていない状態）
const DEFAULT_KEY_INPUT = 0x2FFF;

// keyInputからpressedKeys配列を生成
function getPressedKeys(keyInput: number): KeyName[] {
  const pressed: KeyName[] = [];
  // keyInputはXOR 0x2FFFされた値なので、元に戻す
  const rawValue = keyInput ^ DEFAULT_KEY_INPUT;
  
  for (const [key, bit] of Object.entries(KEY_BITS)) {
    // ビットが0の場合、押されている
    if ((rawValue & (1 << bit)) === 0) {
      pressed.push(key as KeyName);
    }
  }
  return pressed;
}

// pressedKeys配列からkeyInputを生成
function calculateKeyInput(pressedKeys: KeyName[]): number {
  let rawValue = DEFAULT_KEY_INPUT; // すべて1で開始
  
  for (const key of pressedKeys) {
    const bit = KEY_BITS[key];
    // 押されたキーのビットを0にする
    rawValue &= ~(1 << bit);
  }
  
  // XOR 0x2FFFして保存
  return rawValue ^ DEFAULT_KEY_INPUT;
}

export function KeyInputParam() {
  const { searchConditions, setSearchConditions } = useAppStore();
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  
  // 現在押されているキーのリスト
  const pressedKeys = getPressedKeys(searchConditions.keyInput);
  
  const handleToggleKey = (key: KeyName) => {
    const currentPressed = getPressedKeys(searchConditions.keyInput);
    let newPressed: KeyName[];
    
    if (currentPressed.includes(key)) {
      // キーを外す
      newPressed = currentPressed.filter(k => k !== key);
    } else {
      // キーを追加
      newPressed = [...currentPressed, key];
    }
    
    const newKeyInput = calculateKeyInput(newPressed);
    setSearchConditions({ keyInput: newKeyInput });
  };

  const handleReset = () => {
    setSearchConditions({ keyInput: DEFAULT_KEY_INPUT });
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-sm font-medium">Key Input</div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setIsDialogOpen(true)}
          className="gap-2"
        >
          <GameController size={16} />
          Configure
        </Button>
      </div>
      
      {pressedKeys.length > 0 && (
        <div className="text-xs text-muted-foreground">
          Available keys: {pressedKeys.join(', ')}
        </div>
      )}
      
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Key Input Configuration</DialogTitle>
            <DialogDescription>
              Select the keys that can be pressed during seed generation. Layout matches the actual controller.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-6 py-4">
            {/* コントローラレイアウト */}
            <div className="grid grid-cols-3 gap-4">
              {/* 左エリア: 十字キー */}
              <div className="flex flex-col items-center justify-center space-y-2">
                <div className="text-xs font-medium mb-2">D-Pad</div>
                <div className="grid grid-cols-3 gap-1">
                  <div></div>
                  <Toggle
                    value="Up"
                    aria-label="Up"
                    pressed={pressedKeys.includes('Up')}
                    onPressedChange={() => handleToggleKey('Up')}
                    className="w-12 h-12"
                  >
                    Up
                  </Toggle>
                  <div></div>
                  
                  <Toggle
                    value="Left"
                    aria-label="Left"
                    pressed={pressedKeys.includes('Left')}
                    onPressedChange={() => handleToggleKey('Left')}
                    className="w-12 h-12"
                  >
                    Left
                  </Toggle>
                  <div className="w-12 h-12"></div>
                  <Toggle
                    value="Right"
                    aria-label="Right"
                    pressed={pressedKeys.includes('Right')}
                    onPressedChange={() => handleToggleKey('Right')}
                    className="w-12 h-12"
                  >
                    Right
                  </Toggle>
                  
                  <div></div>
                  <Toggle
                    value="Down"
                    aria-label="Down"
                    pressed={pressedKeys.includes('Down')}
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
                <div className="text-xs font-medium mb-2">Center</div>
                <div className="flex gap-2">
                  <Toggle
                    value="Select"
                    aria-label="Select"
                    pressed={pressedKeys.includes('Select')}
                    onPressedChange={() => handleToggleKey('Select')}
                    className="px-3 py-2"
                  >
                    Select
                  </Toggle>
                  <Toggle
                    value="Start"
                    aria-label="Start"
                    pressed={pressedKeys.includes('Start')}
                    onPressedChange={() => handleToggleKey('Start')}
                    className="px-3 py-2"
                  >
                    Start
                  </Toggle>
                </div>
              </div>
              
              {/* 右エリア: A/B/X/Y */}
              <div className="flex flex-col items-center justify-center space-y-2">
                <div className="text-xs font-medium mb-2">Buttons</div>
                <div className="grid grid-cols-3 gap-1">
                  <div></div>
                  <Toggle
                    value="X"
                    aria-label="X"
                    pressed={pressedKeys.includes('X')}
                    onPressedChange={() => handleToggleKey('X')}
                    className="w-12 h-12"
                  >
                    X
                  </Toggle>
                  <div></div>
                  
                  <Toggle
                    value="Y"
                    aria-label="Y"
                    pressed={pressedKeys.includes('Y')}
                    onPressedChange={() => handleToggleKey('Y')}
                    className="w-12 h-12"
                  >
                    Y
                  </Toggle>
                  <div className="w-12 h-12"></div>
                  <Toggle
                    value="A"
                    aria-label="A"
                    pressed={pressedKeys.includes('A')}
                    onPressedChange={() => handleToggleKey('A')}
                    className="w-12 h-12"
                  >
                    A
                  </Toggle>
                  
                  <div></div>
                  <Toggle
                    value="B"
                    aria-label="B"
                    pressed={pressedKeys.includes('B')}
                    onPressedChange={() => handleToggleKey('B')}
                    className="w-12 h-12"
                  >
                    B
                  </Toggle>
                  <div></div>
                </div>
              </div>
            </div>
            
            {/* L/Rボタン */}
            <div className="flex justify-between px-8">
              <Toggle
                value="L"
                aria-label="L"
                pressed={pressedKeys.includes('L')}
                onPressedChange={() => handleToggleKey('L')}
                className="px-6 py-2"
              >
                L
              </Toggle>
              <Toggle
                value="R"
                aria-label="R"
                pressed={pressedKeys.includes('R')}
                onPressedChange={() => handleToggleKey('R')}
                className="px-6 py-2"
              >
                R
              </Toggle>
            </div>
            
            {/* リセットボタン */}
            <div className="flex justify-between items-center pt-4 border-t">
              <div className="text-sm text-muted-foreground">
                Selected: {pressedKeys.length} key{pressedKeys.length !== 1 ? 's' : ''}
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={handleReset}
              >
                Reset All
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

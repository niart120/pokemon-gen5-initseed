import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Toggle } from '@/components/ui/toggle';
import { useAppStore } from '../../../../store/app-store';
import { GameController } from '@phosphor-icons/react';
import { KEY_INPUT_DEFAULT, keyMaskToNames, keyNamesToMask, type KeyName } from '@/lib/utils/key-input';

export function KeyInputParam() {
  const { searchConditions, setSearchConditions } = useAppStore();
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  // 初期状態は全解除（すべてのキーを未選択）にする
  const [tempKeyInput, setTempKeyInput] = useState(KEY_INPUT_DEFAULT);
  
  // 現在利用可能なキーのリスト
  const availableKeys = keyMaskToNames(searchConditions.keyInput);
  const tempAvailableKeys = keyMaskToNames(tempKeyInput);
  
  const handleToggleKey = (key: KeyName) => {
    const currentAvailable = keyMaskToNames(tempKeyInput);
    let newAvailable: KeyName[];
    
    if (currentAvailable.includes(key)) {
      // キーを外す
      newAvailable = currentAvailable.filter(k => k !== key);
    } else {
      // キーを追加
      newAvailable = [...currentAvailable, key];
    }
    
    const newKeyInput = keyNamesToMask(newAvailable);
    setTempKeyInput(newKeyInput);
  };

  const handleReset = () => {
    setTempKeyInput(KEY_INPUT_DEFAULT);
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

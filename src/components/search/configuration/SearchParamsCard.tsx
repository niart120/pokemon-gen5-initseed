import React from 'react';
import { Card } from '@/components/ui/card';
import { StandardCardHeader, StandardCardContent } from '@/components/ui/card-helpers';
import { Separator } from '@/components/ui/separator';
import { Gear } from '@phosphor-icons/react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Toggle } from '@/components/ui/toggle';
import { useAppStore } from '@/store/app-store';
import { GameController } from '@phosphor-icons/react';
import { KEY_INPUT_DEFAULT, keyMaskToNames, keyNamesToMask, type KeyName } from '@/lib/utils/key-input';

export function SearchParamsCard() {
  const { searchConditions, setSearchConditions } = useAppStore();
  const [isDialogOpen, setIsDialogOpen] = React.useState(false);
  const [tempKeyInput, setTempKeyInput] = React.useState(KEY_INPUT_DEFAULT);

  const formatDate = (year: number, month: number, day: number): string => {
    return `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
  };

  const parseDate = (dateString: string) => {
    const date = new Date(dateString);
    return {
      year: date.getFullYear(),
      month: date.getMonth() + 1,
      day: date.getDate(),
    };
  };

  const startDate = formatDate(
    searchConditions.dateRange.startYear,
    searchConditions.dateRange.startMonth,
    searchConditions.dateRange.startDay,
  );

  const endDate = formatDate(
    searchConditions.dateRange.endYear,
    searchConditions.dateRange.endMonth,
    searchConditions.dateRange.endDay,
  );

  const handleStartDateChange = (dateString: string) => {
    if (!dateString) return;
    const { year, month, day } = parseDate(dateString);
    setSearchConditions({
      dateRange: {
        ...searchConditions.dateRange,
        startYear: year,
        startMonth: month,
        startDay: day,
        startHour: 0,
        startMinute: 0,
        startSecond: 0,
      },
    });
  };

  const handleEndDateChange = (dateString: string) => {
    if (!dateString) return;
    const { year, month, day } = parseDate(dateString);
    setSearchConditions({
      dateRange: {
        ...searchConditions.dateRange,
        endYear: year,
        endMonth: month,
        endDay: day,
        endHour: 23,
        endMinute: 59,
        endSecond: 59,
      },
    });
  };

  const availableKeys = React.useMemo(() => keyMaskToNames(searchConditions.keyInput), [searchConditions.keyInput]);
  const tempAvailableKeys = React.useMemo(() => keyMaskToNames(tempKeyInput), [tempKeyInput]);

  const handleToggleKey = (key: KeyName) => {
    const current = keyMaskToNames(tempKeyInput);
    const next = current.includes(key)
      ? current.filter((item) => item !== key)
      : [...current, key];
    setTempKeyInput(keyNamesToMask(next));
  };

  const handleResetKeys = () => {
    setTempKeyInput(KEY_INPUT_DEFAULT);
  };

  const handleApplyKeys = () => {
    setSearchConditions({ keyInput: tempKeyInput });
    setIsDialogOpen(false);
  };

  const openKeyDialog = () => {
    setTempKeyInput(searchConditions.keyInput);
    setIsDialogOpen(true);
  };

  return (
    <Card className="py-2 flex flex-col h-full gap-2">
      <StandardCardHeader icon={<Gear size={20} className="opacity-80" />} title="Search Filters" />
      <StandardCardContent>
        <div className="space-y-3">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="start-date" className="text-sm font-medium">Start Date</Label>
              <Input
                id="start-date"
                type="date"
                min="2000-01-01"
                max="2099-12-31"
                className="h-9"
                value={startDate}
                onChange={(event) => handleStartDateChange(event.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="end-date" className="text-sm font-medium">End Date</Label>
              <Input
                id="end-date"
                type="date"
                min="2000-01-01"
                max="2099-12-31"
                className="h-9"
                value={endDate}
                onChange={(event) => handleEndDateChange(event.target.value)}
              />
            </div>
          </div>
          <div className="text-xs text-muted-foreground">
            Current range: {startDate} to {endDate}
          </div>
          <Separator />
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">Key Input</div>
              <Button variant="outline" size="sm" onClick={openKeyDialog} className="gap-2">
                <GameController size={16} />
                Configure
              </Button>
            </div>
            {availableKeys.length > 0 && (
              <div className="text-xs text-muted-foreground">
                {availableKeys.join(', ')}
              </div>
            )}
          </div>
        </div>
      </StandardCardContent>
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Key Input Configuration</DialogTitle>
          </DialogHeader>
          <div className="space-y-6 py-4">
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
            <div className="grid grid-cols-3 gap-4">
              <div className="flex flex-col items-center justify-center space-y-2">
                <div className="grid grid-cols-3 gap-1 font-arrows">
                  <div />
                  <Toggle
                    value="[↑]"
                    aria-label="Up"
                    pressed={tempAvailableKeys.includes('[↑]')}
                    onPressedChange={() => handleToggleKey('[↑]')}
                    className="w-12 h-12"
                  >
                    [↑]
                  </Toggle>
                  <div />
                  <Toggle
                    value="[←]"
                    aria-label="Left"
                    pressed={tempAvailableKeys.includes('[←]')}
                    onPressedChange={() => handleToggleKey('[←]')}
                    className="w-12 h-12"
                  >
                    [←]
                  </Toggle>
                  <div className="w-12 h-12" />
                  <Toggle
                    value="[→]"
                    aria-label="Right"
                    pressed={tempAvailableKeys.includes('[→]')}
                    onPressedChange={() => handleToggleKey('[→]')}
                    className="w-12 h-12"
                  >
                    [→]
                  </Toggle>
                  <div />
                  <Toggle
                    value="[↓]"
                    aria-label="Down"
                    pressed={tempAvailableKeys.includes('[↓]')}
                    onPressedChange={() => handleToggleKey('[↓]')}
                    className="w-12 h-12"
                  >
                    [↓]
                  </Toggle>
                  <div />
                </div>
              </div>
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
              <div className="flex flex-col items-center justify-center space-y-2">
                <div className="grid grid-cols-3 gap-1">
                  <div />
                  <Toggle
                    value="X"
                    aria-label="X"
                    pressed={tempAvailableKeys.includes('X')}
                    onPressedChange={() => handleToggleKey('X')}
                    className="w-12 h-12"
                  >
                    X
                  </Toggle>
                  <div />
                  <Toggle
                    value="Y"
                    aria-label="Y"
                    pressed={tempAvailableKeys.includes('Y')}
                    onPressedChange={() => handleToggleKey('Y')}
                    className="w-12 h-12"
                  >
                    Y
                  </Toggle>
                  <div className="w-12 h-12" />
                  <Toggle
                    value="A"
                    aria-label="A"
                    pressed={tempAvailableKeys.includes('A')}
                    onPressedChange={() => handleToggleKey('A')}
                    className="w-12 h-12"
                  >
                    A
                  </Toggle>
                  <div />
                  <Toggle
                    value="B"
                    aria-label="B"
                    pressed={tempAvailableKeys.includes('B')}
                    onPressedChange={() => handleToggleKey('B')}
                    className="w-12 h-12"
                  >
                    B
                  </Toggle>
                  <div />
                </div>
              </div>
            </div>
            <div className="flex justify-between items-center pt-4 border-t">
              <Button variant="outline" size="sm" onClick={handleResetKeys}>
                Reset All
              </Button>
              <Button size="sm" onClick={handleApplyKeys}>
                Apply
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </Card>
  );
}

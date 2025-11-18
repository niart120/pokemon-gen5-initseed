import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { MagnifyingGlass, Info, Sparkle } from '@phosphor-icons/react';
import { GenerationPanel } from './GenerationPanel';
import { useAppStore } from '@/store/app-store';
import { useResponsiveLayout } from '@/hooks/use-mobile';
import { SearchPanel } from './SearchPanel';
import { HelpPanel } from './HelpPanel';

export function MainContent() {
  const { activeTab, setActiveTab } = useAppStore();
  const { isStack } = useResponsiveLayout();

  const overflowClasses = isStack ? 'overflow-y-auto overflow-x-hidden' : 'overflow-y-auto overflow-x-auto';
  const layoutClasses = 'flex flex-col';

  return (
    <main className={`px-2 sm:px-3 lg:px-3 xl:px-4 2xl:px-4 py-1 max-w-none flex-1 min-h-0 ${layoutClasses} ${overflowClasses}`}>
      <div className="max-w-screen-2xl xl:max-w-[1700px] 2xl:max-w-[1900px] mx-auto w-full flex-1 flex flex-col min-w-0 min-h-0 gap-3">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-2 flex flex-col flex-1 min-h-0">
          <TabsList className="grid grid-cols-3 w-full max-w-6xl mx-auto flex-shrink-0 h-9">
            <TabsTrigger value="search" className="flex items-center gap-2">
              <MagnifyingGlass size={16} />
              Search
            </TabsTrigger>
            <TabsTrigger value="generation" className="flex items-center gap-2">
              <Sparkle size={16} />
              Generation
            </TabsTrigger>
            <TabsTrigger value="help" className="flex items-center gap-2">
              <Info size={16} />
              Help
            </TabsTrigger>
          </TabsList>

          <TabsContent value="search" className="flex-1 min-h-0 overflow-hidden">
            <SearchPanel />
          </TabsContent>

          <TabsContent value="generation" className="flex-1 min-h-0 overflow-hidden">
            <GenerationPanel />
          </TabsContent>

          <TabsContent value="help" className="flex-1 min-h-0 overflow-hidden">
            <HelpPanel />
          </TabsContent>
        </Tabs>
      </div>
    </main>
  );
}

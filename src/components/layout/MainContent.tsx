import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { MagnifyingGlass, Info, Sparkle, Egg, Wrench } from '@phosphor-icons/react';
import { GenerationPanel } from './GenerationPanel';
import { EggGenerationPanel } from './EggGenerationPanel';
import { EggSearchPanel } from '@/components/egg-search';
import { MiscPanel } from '@/components/misc';
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
          <div className="overflow-x-auto flex-shrink-0 -mx-2 px-2 scrollbar-thin scrollbar-thumb-muted scrollbar-track-transparent">
            <TabsList className="inline-flex w-max min-w-full md:grid md:grid-cols-6 md:w-full max-w-6xl mx-auto h-9 gap-1 md:gap-0">
              <TabsTrigger value="search" className="flex items-center gap-2 whitespace-nowrap px-3 md:px-2">
                <MagnifyingGlass size={16} />
                Search
              </TabsTrigger>
              <TabsTrigger value="generation" className="flex items-center gap-2 whitespace-nowrap px-3 md:px-2">
                <Sparkle size={16} />
                Generation
              </TabsTrigger>
              <TabsTrigger value="egg-search" className="flex items-center gap-2 whitespace-nowrap px-3 md:px-2">
                <Egg size={16} />
                Search(Egg)
              </TabsTrigger>
              <TabsTrigger value="egg" className="flex items-center gap-2 whitespace-nowrap px-3 md:px-2">
                <Egg size={16} />
                Generation(Egg)
              </TabsTrigger>
              <TabsTrigger value="misc" className="flex items-center gap-2 whitespace-nowrap px-3 md:px-2">
                <Wrench size={16} />
                Misc
              </TabsTrigger>
              <TabsTrigger value="help" className="flex items-center gap-2 whitespace-nowrap px-3 md:px-2">
                <Info size={16} />
                Help
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="search" className="flex-1 min-h-0 overflow-hidden">
            <SearchPanel />
          </TabsContent>

          <TabsContent value="generation" className="flex-1 min-h-0 overflow-hidden">
            <GenerationPanel />
          </TabsContent>

          <TabsContent value="egg-search" className="flex-1 min-h-0 overflow-hidden">
            <EggSearchPanel />
          </TabsContent>

          <TabsContent value="egg" className="flex-1 min-h-0 overflow-hidden">
            <EggGenerationPanel />
          </TabsContent>

          <TabsContent value="misc" className="flex-1 min-h-0 overflow-hidden">
            <MiscPanel />
          </TabsContent>

          <TabsContent value="help" className="flex-1 min-h-0 overflow-hidden">
            <HelpPanel />
          </TabsContent>
        </Tabs>
      </div>
    </main>
  );
}

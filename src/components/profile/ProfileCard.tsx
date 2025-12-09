import React from 'react';
import { PanelCard } from '@/components/ui/panel-card';
import { Separator } from '@/components/ui/separator';
import { DeviceMobileSpeaker } from '@phosphor-icons/react';
import type { Hardware, ROMRegion, ROMVersion } from '@/types/rom';
import { useProfileCardForm } from '@/components/profile/hooks/useProfileCardForm';
import { CollapsedSection } from '@/components/profile/sections/CollapsedSection';
import { RomHardwareSection } from '@/components/profile/sections/RomHardwareSection';
import { Timer0VCountSection } from '@/components/profile/sections/Timer0VCountSection';
import { GameStateSection } from '@/components/profile/sections/GameStateSection';
import { ProfileManagementSection } from '@/components/profile/sections/ProfileManagementSection';
import { ProfileErrorsAlert } from '@/components/profile/sections/ProfileErrorsAlert';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import { profileCardTitle, resolveProfileSectionTitle } from '@/lib/i18n/strings/profile-card';

const ROM_VERSIONS: ROMVersion[] = ['B', 'W', 'B2', 'W2'];
const ROM_REGIONS: ROMRegion[] = ['JPN', 'KOR', 'USA', 'GER', 'FRA', 'SPA', 'ITA'];
const HARDWARE_OPTIONS: { value: Hardware; label: string }[] = [
  { value: 'DS', label: 'DS' },
  { value: 'DS_LITE', label: 'DS Lite' },
];

export function ProfileCard() {
  const { errors, profileSelection, header, layout, rom, timer, game } = useProfileCardForm();
  const locale = useLocale();

  const { profiles, activeId, onSelect } = profileSelection;
  const { profileName, canModify, disableDelete, lockedReason, onProfileNameChange, onDelete } = header;
  const { sectionOpen, isStack, toggleSection } = layout;

  const profileManagementSection = (
    <ProfileManagementSection
      profiles={profiles}
      activeProfileId={activeId}
      profileName={profileName}
      canModify={canModify}
      disableDelete={disableDelete}
        lockedReason={lockedReason}
      onSelectProfile={onSelect}
      onProfileNameChange={onProfileNameChange}
      onDelete={onDelete}
    />
  );

  const romSection = (
    <RomHardwareSection
      romVersion={rom.romVersion}
      romRegion={rom.romRegion}
      hardware={rom.hardware}
      macSegments={rom.macSegments}
      romVersions={ROM_VERSIONS}
      romRegions={ROM_REGIONS}
      hardwareOptions={HARDWARE_OPTIONS}
      macInputRefs={rom.macInputRefs}
      onRomVersionChange={rom.onRomVersionChange}
      onRomRegionChange={rom.onRomRegionChange}
      onHardwareChange={rom.onHardwareChange}
      onMacSegmentChange={rom.onMacSegmentChange}
      onMacSegmentFocus={rom.onMacSegmentFocus}
      onMacSegmentMouseDown={rom.onMacSegmentMouseDown}
      onMacSegmentClick={rom.onMacSegmentClick}
      onMacSegmentKeyDown={rom.onMacSegmentKeyDown}
        onMacSegmentPaste={rom.onMacSegmentPaste}
        disabled={rom.disabled}
    />
  );

  const timerSection = (
    <Timer0VCountSection
      timer0Auto={timer.timer0Auto}
      timer0Min={timer.timer0Min}
      timer0Max={timer.timer0Max}
      vcountMin={timer.vcountMin}
      vcountMax={timer.vcountMax}
      onAutoToggle={timer.onAutoToggle}
      onTimerHexChange={timer.onTimerHexChange}
      onTimerHexBlur={timer.onTimerHexBlur}
      onVCountHexChange={timer.onVCountHexChange}
        onVCountHexBlur={timer.onVCountHexBlur}
        disabled={timer.disabled}
    />
  );

  const gameSection = (
    <GameStateSection
      tid={game.tid}
      sid={game.sid}
      newGame={game.newGame}
      withSave={game.withSave}
      shinyCharm={game.shinyCharm}
      memoryLink={game.memoryLink}
      onTidChange={game.onTidChange}
      onTidBlur={game.onTidBlur}
      onSidChange={game.onSidChange}
      onSidBlur={game.onSidBlur}
      onNewGameToggle={game.onNewGameToggle}
      onWithSaveToggle={game.onWithSaveToggle}
      onShinyCharmToggle={game.onShinyCharmToggle}
        onMemoryLinkToggle={game.onMemoryLinkToggle}
        withSaveDisabled={game.withSaveDisabled}
        shinyCharmDisabled={game.shinyCharmDisabled}
        memoryLinkDisabled={game.memoryLinkDisabled}
        disabled={game.disabled}
    />
  );

  return (
    <PanelCard
      icon={<DeviceMobileSpeaker size={20} className="opacity-80" />}
      title={resolveLocaleValue(profileCardTitle, locale)}
      fullHeight={false}
    >
        {errors.length > 0 && (
          <>
            <ProfileErrorsAlert errors={errors} />
            <Separator />
          </>
        )}

        <div className="grid gap-2 lg:grid-cols-4">
          {isStack ? (
            <>
              <CollapsedSection
                sectionKey="profile"
                headingId="profile-management"
                title={resolveProfileSectionTitle('profileManagement', locale)}
                isOpen={sectionOpen.profile}
                onToggle={toggleSection}
              >
                {profileManagementSection}
              </CollapsedSection>
              <CollapsedSection
                sectionKey="rom"
                headingId="profile-rom"
                title={resolveProfileSectionTitle('romHardware', locale)}
                isOpen={sectionOpen.rom}
                onToggle={toggleSection}
              >
                {romSection}
              </CollapsedSection>
              <CollapsedSection
                sectionKey="timer"
                headingId="profile-timer"
                title={resolveProfileSectionTitle('timer0Vcount', locale)}
                isOpen={sectionOpen.timer}
                onToggle={toggleSection}
              >
                {timerSection}
              </CollapsedSection>
              <CollapsedSection
                sectionKey="game"
                headingId="profile-game"
                title={resolveProfileSectionTitle('gameState', locale)}
                isOpen={sectionOpen.game}
                onToggle={toggleSection}
              >
                {gameSection}
              </CollapsedSection>
            </>
          ) : (
            <>
              <section className="space-y-3" aria-labelledby="profile-management">
                <h4 id="profile-management" className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  {resolveProfileSectionTitle('profileManagement', locale)}
                </h4>
                {profileManagementSection}
              </section>
              <section className="space-y-3" aria-labelledby="profile-rom">
                <h4 id="profile-rom" className="text-xs font-semibold text-muted-foreground tracking-wide uppercase">
                  {resolveProfileSectionTitle('romHardware', locale)}
                </h4>
                {romSection}
              </section>
              <section className="space-y-3" aria-labelledby="profile-timer">
                <h4 id="profile-timer" className="text-xs font-semibold text-muted-foreground tracking-wide uppercase">
                  {resolveProfileSectionTitle('timer0Vcount', locale)}
                </h4>
                {timerSection}
              </section>
              <section className="space-y-3" aria-labelledby="profile-game">
                <h4 id="profile-game" className="text-xs font-semibold text-muted-foreground tracking-wide uppercase">
                  {resolveProfileSectionTitle('gameState', locale)}
                </h4>
                {gameSection}
              </section>
            </>
          )}
        </div>
    </PanelCard>
  );
}

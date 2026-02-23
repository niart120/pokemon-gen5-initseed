import type { ReactNode, ComponentPropsWithoutRef } from 'react';
import { Card } from '@/components/ui/card';
import { StandardCardHeader, StandardCardContent } from '@/components/ui/card-helpers';
import { cn } from '@/lib/utils';

type PanelCardScrollMode = 'content' | 'parent';
type PanelCardPadding = 'default' | 'none';
type PanelCardSpacing = 'default' | 'compact' | 'none';

type CardBaseProps = Omit<ComponentPropsWithoutRef<typeof Card>, 'title'>;

interface PanelCardProps extends CardBaseProps {
  icon?: ReactNode;
  title: ReactNode;
  children: ReactNode;
  headerActions?: ReactNode;
  headerClassName?: string;
  contentClassName?: string;
  fullHeight?: boolean;
  scrollMode?: PanelCardScrollMode;
  padding?: PanelCardPadding;
  spacing?: PanelCardSpacing;
}

const spacingClassMap: Record<PanelCardSpacing, string | undefined> = {
  default: undefined,
  compact: 'space-y-1.5',
  none: 'space-y-0',
};

const paddingClassMap: Record<PanelCardPadding, string | undefined> = {
  default: undefined,
  none: 'p-0',
};

export function PanelCard({
  icon,
  title,
  children,
  headerActions,
  headerClassName,
  contentClassName,
  fullHeight = true,
  scrollMode = 'content',
  padding = 'default',
  spacing = 'default',
  className,
  ...cardProps
}: PanelCardProps) {
  return (
    <Card
      className={cn('py-2 flex flex-col gap-2', fullHeight && 'h-full', className)}
      {...cardProps}
    >
      <StandardCardHeader icon={icon} title={title} className={headerClassName} action={headerActions} />
      <StandardCardContent
        className={cn(spacingClassMap[spacing], paddingClassMap[padding], contentClassName)}
        noScroll={scrollMode === 'parent'}
      >
        {children}
      </StandardCardContent>
    </Card>
  );
}

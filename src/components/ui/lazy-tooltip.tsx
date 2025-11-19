import { useState } from 'react';
import { Tooltip, TooltipContent, TooltipTrigger } from './tooltip';

interface LazyTooltipProps {
  trigger: React.ReactNode;
  renderContent: () => React.ReactNode;
  side?: 'top' | 'right' | 'bottom' | 'left';
  className?: string;
}

/**
 * Lazy loading tooltip component that only renders content on first hover.
 * Optimizes memory usage for large lists by deferring expensive computations.
 */
export function LazyTooltip({ trigger, renderContent, side = 'bottom', className }: LazyTooltipProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [hasRendered, setHasRendered] = useState(false);

  const handleOpenChange = (open: boolean) => {
    setIsOpen(open);
    if (open && !hasRendered) {
      setHasRendered(true);
    }
  };

  return (
    <Tooltip open={isOpen} onOpenChange={handleOpenChange}>
      <TooltipTrigger asChild>{trigger}</TooltipTrigger>
      {hasRendered && (
        <TooltipContent side={side} className={className}>
          {renderContent()}
        </TooltipContent>
      )}
    </Tooltip>
  );
}

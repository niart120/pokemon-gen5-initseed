// Shared card style helpers: カードタイトル/ヘッダ/コンテンツの繰り返しクラスを一元化
import { cn } from '@/lib/utils';
import { CardHeader, CardTitle, CardContent } from './card';
import type { ReactNode } from 'react';

interface StandardCardHeaderProps {
  icon?: ReactNode;
  title: ReactNode; // string または装飾用ノード (a11y用に span など)
  className?: string;
}
export function StandardCardHeader({ icon, title, className }: StandardCardHeaderProps) {
  return (
    <CardHeader className={cn('pb-0', className)}>
      <CardTitle className="text-base flex items-center gap-2">
        {icon}
        {title}
      </CardTitle>
    </CardHeader>
  );
}

interface StandardCardContentProps {
  children: ReactNode;
  className?: string;
}
export function StandardCardContent({ children, className }: StandardCardContentProps) {
  return <CardContent className={cn('space-y-2 flex-1 min-h-0 flex flex-col overflow-y-auto', className)}>{children}</CardContent>;
}

// 単純なメトリクスグリッド: ラベル + 値 (mono)
interface MetricItem { label: string; value: ReactNode; }
interface MetricsGridProps { items: MetricItem[]; columns?: string; className?: string; }
export function MetricsGrid({ items, columns = 'grid-cols-2 md:grid-cols-6', className }: MetricsGridProps) {
  return (
    <div className={cn('grid gap-2 text-xs', columns, className)}>
      {items.map((it, i) => (
        <div key={i} className="space-y-0.5">
          <div className="text-muted-foreground">{it.label}</div>
          <div className="font-mono text-sm leading-tight">{it.value}</div>
        </div>
      ))}
    </div>
  );
}

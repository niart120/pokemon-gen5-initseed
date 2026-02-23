import * as React from "react"

import { cn } from "@/lib/utils"

const Table = React.forwardRef<HTMLTableElement, React.ComponentPropsWithoutRef<"table">>(
  ({ className, ...props }, ref) => {
    return (
      <table
        data-slot="table"
        ref={ref}
        className={cn("w-full caption-bottom text-sm", className)}
        {...props}
      />
    )
  }
)
Table.displayName = "Table"

const TableHeader = React.forwardRef<HTMLTableSectionElement, React.ComponentPropsWithoutRef<"thead">>(
  ({ className, ...props }, ref) => (
    <thead
      data-slot="table-header"
      ref={ref}
      className={cn("[&_tr]:border-b", className)}
      {...props}
    />
  )
)
TableHeader.displayName = "TableHeader"

const TableBody = React.forwardRef<HTMLTableSectionElement, React.ComponentPropsWithoutRef<"tbody">>(
  ({ className, ...props }, ref) => (
    <tbody
      data-slot="table-body"
      ref={ref}
      className={cn("[&_tr:last-child]:border-0", className)}
      {...props}
    />
  )
)
TableBody.displayName = "TableBody"

const TableFooter = React.forwardRef<HTMLTableSectionElement, React.ComponentPropsWithoutRef<"tfoot">>(
  ({ className, ...props }, ref) => (
    <tfoot
      data-slot="table-footer"
      ref={ref}
      className={cn(
        "bg-muted/50 border-t font-medium [&>tr]:last:border-b-0",
        className
      )}
      {...props}
    />
  )
)
TableFooter.displayName = "TableFooter"

const TableRow = React.forwardRef<HTMLTableRowElement, React.ComponentPropsWithoutRef<"tr">>(
  ({ className, ...props }, ref) => (
    <tr
      data-slot="table-row"
      ref={ref}
      className={cn(
        "hover:bg-muted/50 data-[state=selected]:bg-muted border-b transition-colors",
        className
      )}
      {...props}
    />
  )
)
TableRow.displayName = "TableRow"

const TableHead = React.forwardRef<HTMLTableCellElement, React.ComponentPropsWithoutRef<"th">>(
  ({ className, ...props }, ref) => (
    <th
      data-slot="table-head"
      ref={ref}
      className={cn(
        "text-foreground h-10 px-2 text-left align-middle font-medium [&:has([role=checkbox])]:pr-0 [&>[role=checkbox]]:translate-y-[2px]",
        className
      )}
      {...props}
    />
  )
)
TableHead.displayName = "TableHead"

const TableCell = React.forwardRef<HTMLTableCellElement, React.ComponentPropsWithoutRef<"td">>(
  ({ className, ...props }, ref) => (
    <td
      data-slot="table-cell"
      ref={ref}
      className={cn(
        "p-2 align-middle [&:has([role=checkbox])]:pr-0 [&>[role=checkbox]]:translate-y-[2px]",
        className
      )}
      {...props}
    />
  )
)
TableCell.displayName = "TableCell"

const TableCaption = React.forwardRef<HTMLTableCaptionElement, React.ComponentPropsWithoutRef<"caption">>(
  ({ className, ...props }, ref) => (
    <caption
      data-slot="table-caption"
      ref={ref}
      className={cn("text-muted-foreground mt-4 text-sm", className)}
      {...props}
    />
  )
)
TableCaption.displayName = "TableCaption"

export {
  Table,
  TableHeader,
  TableBody,
  TableFooter,
  TableHead,
  TableRow,
  TableCell,
  TableCaption,
}

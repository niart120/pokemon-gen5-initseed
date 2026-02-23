import React from 'react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { useLocale } from '@/lib/i18n/locale-context';
import { resolveLocaleValue } from '@/lib/i18n/strings/types';
import {
  deprecationNoticeTitle,
  deprecationNoticeBody,
  deprecationNoticeNewVersion,
  deprecationNoticeGuide,
  deprecationNoticeAppLabel,
  deprecationNoticeArticleLabel,
  deprecationNoticeArticleTitle,
  deprecationNoticeClose,
} from '@/lib/i18n/strings/deprecation-notice';

/**
 * アプリが非推奨であることをユーザに通知するダイアログ。
 * ページ訪問時に自動表示し、新バージョンへの誘導を行う。
 */
export function DeprecationNoticeDialog() {
  const [open, setOpen] = React.useState(true);
  const locale = useLocale();

  return (
    <AlertDialog open={open} onOpenChange={setOpen}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>
            {resolveLocaleValue(deprecationNoticeTitle, locale)}
          </AlertDialogTitle>
          <AlertDialogDescription asChild>
            <div className="space-y-3 text-sm text-muted-foreground">
              <p>
                {resolveLocaleValue(deprecationNoticeBody, locale)}
              </p>
              <p>
                {resolveLocaleValue(deprecationNoticeNewVersion, locale)}
              </p>
              <p>
                {resolveLocaleValue(deprecationNoticeGuide, locale)}
              </p>
              <ul className="list-disc list-inside space-y-1">
                <li>
                  {resolveLocaleValue(deprecationNoticeAppLabel, locale)}{' '}
                  <a
                    href="https://niart120.github.io/5genSearch-web/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-semibold text-primary underline underline-offset-4 hover:text-primary/80"
                  >
                    5genSearch-web
                  </a>
                </li>
                <li>
                  {resolveLocaleValue(deprecationNoticeArticleLabel, locale)}{' '}
                  <a
                    href="https://hackmd.io/@niart/rJ3NkfdObg"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-semibold text-primary underline underline-offset-4 hover:text-primary/80"
                  >
                    {resolveLocaleValue(deprecationNoticeArticleTitle, locale)}
                  </a>
                </li>
              </ul>
            </div>
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogAction onClick={() => setOpen(false)}>
            {resolveLocaleValue(deprecationNoticeClose, locale)}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

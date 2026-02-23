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

/**
 * アプリが非推奨であることをユーザに通知するダイアログ。
 * 初回訪問時に自動表示し、新バージョンへの誘導を行う。
 */
export function DeprecationNoticeDialog() {
  const [open, setOpen] = React.useState(true);

  return (
    <AlertDialog open={open} onOpenChange={setOpen}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>メンテナンス終了のお知らせ</AlertDialogTitle>
          <AlertDialogDescription asChild>
            <div className="space-y-3 text-sm text-muted-foreground">
              <p>
                本アプリは現在メンテナンスされていません。
              </p>
              <p>
                機能・UI・内部構成を全面的に見直した新バージョンのアプリを公開しています。
                新規利用の方は以下をご覧ください。
              </p>
              <ul className="list-disc list-inside space-y-1">
                <li>
                  アプリ:{' '}
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
                  記事:{' '}
                  <a
                    href="https://hackmd.io/@niart/rJ3NkfdObg"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-semibold text-primary underline underline-offset-4 hover:text-primary/80"
                  >
                    第五世代乱数調整webアプリを再構築した話
                  </a>
                </li>
              </ul>
            </div>
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogAction onClick={() => setOpen(false)}>
            閉じる
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

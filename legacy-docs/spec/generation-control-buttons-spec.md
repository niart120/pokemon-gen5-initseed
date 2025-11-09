# Generation Control Buttons Spec (Task 7)

目的: GenerationControlCard のボタン状態管理 / variant / アイコン / アクセシビリティを SearchControlCard と整合させる。

## 1. 現状比較
| 状態 | Search (現行) | Generation (現行) |
|------|---------------|-------------------|
| Idle (未開始 / 完了後) | Start Search (primary, icon Play) 有効 (ターゲットが無いと disabled) | Start (green手製) Enabled, Pause/Resume/Stop も常時表示 (一部 disabled) |
| Starting | Start Search disabled (== isRunning true 前提 初期瞬間) | Start disabled (status==='starting') 他ボタン表示 |
| Running | Pause (secondary, icon Pause) + Stop (destructive, icon Square) | Start disabled, Pause enabled(yellow), Resume disabled, Stop enabled(red) 常時ボタン列 |
| Paused | Resume (primary, icon Play) + Stop (destructive) | Start disabled, Pause disabled, Resume enabled(blue), Stop enabled(red) |
| Stopped (完了/手動) | Start Search 再表示 (previous results保持) | 表示は開始時と同じ (全ボタン), disabled 状態変化のみ |

問題: Generation 側は常時4ボタン表示 → 認知負荷/横幅占有。Search 側は状態駆動で表示最小化。Color variant が Tailwind生色 (green/yellow/blue/red) でガイドライン逸脱。

## 2. 状態マシン (Generation)
既存 status 値: 'idle' | 'starting' | 'running' | 'paused' | 'stopped' (推定: コード上 'stopped' or 'completed' 相当は lastCompletion?)

遷移図 (簡潔):
Idle → (Start) → Starting → Running → (Pause) → Paused → (Resume) → Running
Running → (Stop) → Idle (or Completed idle-like)
Paused → (Stop) → Idle
Running → (Complete内部) → Idle (lastCompletion 記録)

## 3. 目標UI状態定義
| status | 表示ボタンセット | 優先アクション配置順 | 備考 |
|--------|------------------|------------------------|------|
| idle / stopped | Start | [Start] | Start: variant=primary (size=sm, icon Play) |
| starting | (非活性表示) Start disabled (spinning?) | [Start(disabled)] | オプション: スケルトンor spinner (後続拡張) |
| running | Pause, Stop | [Pause, Stop] | Pause: variant=secondary, Stop: variant=destructive |
| paused | Resume, Stop | [Resume, Stop] | Resume: variant=primary, Stop: variant=destructive |

補足: Start 再押下時に validationErrors があればエラー領域表示 (現行仕様踏襲)。

## 4. コンポーネント仕様
- Buttonコンポーネント: `@/components/ui/button` 利用
- アイコン: `@phosphor-icons/react` Play / Pause / Square (Search と統一)
- サイズ: `size="sm"`
- レイアウト: Flex行; 一番左の主要ボタンに `flex-1` (Pause/Resume 時) を付与し横幅バランス (Search 参考)。Stop は内容幅最小。
- 余白: `gap-2 flex-wrap`
- Card スタイル統合: `py-2` / Title `text-base flex items-center gap-2` (Play アイコン optional: Generation 側にも付与して整合または未使用→Search に合わせ付与案)

## 5. Validation エラー表示
- 位置: ボタン列上部 (現行維持)
- スタイル: `text-destructive text-xs space-y-0.5` (Search 側での error pattern があれば合わせる; 無ければ destructive color token)
- role="alert" aria-live="polite"

## 6. 型/ロジック
擬似コード:
```tsx
const { status, validateDraft, validationErrors, startGeneration, pauseGeneration, resumeGeneration, stopGeneration } = useAppStore();

const canStart = status === 'idle' || status === 'stopped';
const isStarting = status === 'starting';
const isRunning = status === 'running';
const isPaused  = status === 'paused';

return (
 <div className="flex gap-2 flex-wrap">
  {canStart && (
    <Button size="sm" onClick={handleStart} disabled={isStarting} className="flex-1">
      <Play size={16} className="mr-2"/>Start
    </Button>
  )}
  {isRunning && (
    <>
      <Button size="sm" variant="secondary" onClick={pauseGeneration} className="flex-1">
        <Pause size={16} className="mr-2"/>Pause
      </Button>
      <Button size="sm" variant="destructive" onClick={stopGeneration}>
        <Square size={16} className="mr-2"/>Stop
      </Button>
    </>
  )}
  {isPaused && (
    <>
      <Button size="sm" onClick={resumeGeneration} className="flex-1">
        <Play size={16} className="mr-2"/>Resume
      </Button>
      <Button size="sm" variant="destructive" onClick={stopGeneration}>
        <Square size={16} className="mr-2"/>Stop
      </Button>
    </>
  )}
 </div>
);
```

## 7. アクセシビリティ/フォーカス
- Start/Pause/Resume/Stop はタブ順に状態依存表示 (不要ボタンは DOM から削除) → スクリーンリーダーが不要な無効ボタンを読み上げない
- 状態変更時に focus を主要アクションへ移譲 (例: Pause 押下後 Pause→Resumeに変化したら `ref` 経由で Resume に focus) — オプション実装
- ショートカット拡張余地: Space/Enter で主要ボタン; 後続タスクで検討

## 8. 実装ステップ (Task 8 用)
1. GenerationControlCard 内で Button/Icon import, 既存 button 削除し条件描画実装
2. status 判定ロジック導入 + canStart 等ブール
3. validationErrors スタイル調整 (destructive color)
4. Title text-base & アイコン追加 (Play size=20) ※ガイドライン準拠
5. (任意) starting 状態に Spinner (後回し) — 余地をコメント
6. Vitest snapshot/状態テスト (Task 13) 下準備: 主要要素 data-testid 付与

## 9. data-testid 提案
| 要素 | testid |
|------|--------|
| Start Button | gen-start-btn |
| Pause Button | gen-pause-btn |
| Resume Button | gen-resume-btn |
| Stop Button | gen-stop-btn |
| Error Container | gen-validation-errors |

## 10. リスク/互換性
- DOM からボタンが消えることで外部参照 (もしあれば) 破壊リスク: 現状他コンポーネントから直接参照していない想定
- E2E テスト (Playwright) がボタン text で依存している場合: Pause/Resume の存在タイミングが変わるのでテスト更新必要

---
(本ファイルは Task 7 成果物)

// 共通レイアウト定数
// Generation / Search 両パネルで利用する左カラム基準幅。
// clamp(min, preferred, max) で 2段ブレークポイントを1本化し、
// 将来 uiScale による動的調整を行う場合はここを関数化する想定。
export const LEFT_COLUMN_WIDTH_CLAMP = 'clamp(420px,34vw,560px)';

import { ref, computed, type ComputedRef, type Ref } from 'vue';

export type RailSnakeRow = {
  rowIndex: number;
  /** Odd rows: indices reversed in DOM, still laid out LTR (#8…#5); arrows flow ←. */
  reversed: boolean;
  shotIndices: number[];
};

/** Keyframe tile + inline segment slot width (px). */
export const RAIL_NODE_W_PX = 176;
export const RAIL_SEGMENT_W_PX = 88;
export const RAIL_TURN_W_PX = 72;

const STRIDE_PX = RAIL_NODE_W_PX + RAIL_SEGMENT_W_PX;

export function measureRailItemsPerRow(containerWidth: number): number {
  if (containerWidth <= 0) return 4;
  const budget = containerWidth - 48;
  let k = 1;
  for (;;) {
    const next = k + 1;
    const need = next * RAIL_NODE_W_PX + k * RAIL_SEGMENT_W_PX;
    if (need > budget || next > 99) break;
    k = next;
  }
  return Math.max(2, k);
}

/** Z-snake: odd rows list indices high→low so entry keyframe sits on the right under the turn. */
export function buildRailSnakeRows(shotCount: number, itemsPerRow: number): RailSnakeRow[] {
  if (shotCount <= 0) return [];
  const cap = Math.max(2, itemsPerRow);
  const rows: RailSnakeRow[] = [];
  let start = 0;
  let rowIndex = 0;
  while (start < shotCount) {
    const end = Math.min(start + cap, shotCount);
    const indices = Array.from({ length: end - start }, (_, i) => start + i);
    const reversed = rowIndex % 2 === 1;
    rows.push({
      rowIndex,
      reversed,
      shotIndices: reversed ? [...indices].reverse() : indices,
    });
    start = end;
    rowIndex += 1;
  }
  return rows;
}

export function horizontalEdgeIndex(row: RailSnakeRow, pos: number): number {
  return Math.min(row.shotIndices[pos], row.shotIndices[pos + 1]);
}

/** Edge index for the vertical turn after this row (last story index in row → next keyframe). */
export function rowTurnEdgeIndex(row: RailSnakeRow): number {
  return Math.max(...row.shotIndices);
}

/** Story-exit column: LTR row exits right; reversed row exits left. */
export function exitColumn(row: RailSnakeRow): number {
  const n = row.shotIndices.length;
  if (n <= 0) return 0;
  return row.reversed ? 0 : n - 1;
}

/** Entry column for the first story index in this row. */
export function entryColumn(row: RailSnakeRow): number {
  const n = row.shotIndices.length;
  if (n <= 0) return 0;
  return row.reversed ? n - 1 : 0;
}

/** Leading spacer so this row's entry keyframe aligns with the previous row's turn column. */
export function rowLeadSpacerPx(row: RailSnakeRow, prevRow: RailSnakeRow): number {
  const target = exitColumn(prevRow);
  const entry = entryColumn(row);
  return Math.max(0, (target - entry) * STRIDE_PX);
}

/** Spacer width to align turn chip under the row-exit keyframe column. */
export function turnLaneSpacerPx(row: RailSnakeRow): number {
  const centerInNode = (RAIL_NODE_W_PX - RAIL_TURN_W_PX) / 2;
  return exitColumn(row) * STRIDE_PX + centerInNode;
}

export function useLongVideoRailSnakeLayout(
  shotCount: ComputedRef<number>,
  scrollEl: Ref<HTMLElement | null>,
) {
  const itemsPerRow = ref(4);

  function updateItemsPerRow() {
    const el = scrollEl.value;
    if (!el) return;
    itemsPerRow.value = measureRailItemsPerRow(el.clientWidth);
  }

  const railRows = computed(() => buildRailSnakeRows(shotCount.value, itemsPerRow.value));

  return {
    itemsPerRow,
    railRows,
    updateItemsPerRow,
  };
}

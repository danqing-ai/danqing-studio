/** Format / parse long-video storyboard in a single prompt textarea. */

export function formatStoryboardScript(opening: string, segments: string[]): string {
  const parts: string[] = [];
  const open = opening.trim();
  if (open) parts.push(`[Opening]\n${open}`);
  segments.forEach((seg, i) => {
    const s = seg.trim();
    if (s) parts.push(`[Segment ${i + 1}]\n${s}`);
  });
  return parts.join('\n\n');
}

export function parseStoryboardPrompt(text: string): {
  opening: string;
  segmentPrompts: string[];
  isStructured: boolean;
} {
  const raw = (text || '').trim();
  if (!raw) {
    return { opening: '', segmentPrompts: [], isStructured: false };
  }

  const openingMatch = /\[Opening\]\s*([\s\S]*?)(?=\[Segment\s*\d+\]|\Z)/i.exec(raw);
  const segmentMatches = [...raw.matchAll(/\[Segment\s*\d+\]\s*([\s\S]*?)(?=\[Segment\s*\d+\]|\Z)/gi)];

  if (openingMatch || segmentMatches.length > 0) {
    const opening = (openingMatch?.[1] || '').trim();
    const segmentPrompts = segmentMatches.map((m) => m[1].trim()).filter(Boolean);
    return {
      opening: opening || segmentPrompts.shift() || raw,
      segmentPrompts,
      isStructured: true,
    };
  }

  return { opening: raw, segmentPrompts: [], isStructured: false };
}

/** Target duration (sec) that triggers multi-extend long video on supported models. */
export const LONG_VIDEO_MIN_TARGET_SEC = 30;

export function isLongVideoTargetDuration(sec: number, longVideoSupport: boolean): boolean {
  return Boolean(longVideoSupport) && Number(sec) >= LONG_VIDEO_MIN_TARGET_SEC;
}

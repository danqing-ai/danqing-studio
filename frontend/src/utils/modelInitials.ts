/** Same label truncation as model cards in ModelsView (before first `-` or space). */
export function modelInitialsFromName(name: string, fallback = 'M'): string {
  const label = String(name || '').trim();
  if (!label) return fallback;
  const dashIndex = label.indexOf('-');
  const spaceIndex = label.indexOf(' ');
  let endIndex = -1;
  if (dashIndex !== -1 && spaceIndex !== -1) {
    endIndex = Math.min(dashIndex, spaceIndex);
  } else if (dashIndex !== -1) {
    endIndex = dashIndex;
  } else if (spaceIndex !== -1) {
    endIndex = spaceIndex;
  }
  if (endIndex !== -1) {
    return label.slice(0, endIndex);
  }
  return label.slice(0, 3);
}

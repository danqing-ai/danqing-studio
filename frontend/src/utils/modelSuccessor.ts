const DISMISS_PREFIX = 'dq-studio.successorDismiss.v4.';

export function isSuccessorHintDismissed(modelId: string): boolean {
  if (!modelId) return false;
  try {
    return localStorage.getItem(`${DISMISS_PREFIX}${modelId}`) === '1';
  } catch {
    return false;
  }
}

export function dismissSuccessorHint(modelId: string): void {
  if (!modelId) return;
  try {
    localStorage.setItem(`${DISMISS_PREFIX}${modelId}`, '1');
  } catch {
    /* ignore quota / private mode */
  }
}

export function modelHasSuccessor(model: { successor?: string | null } | null | undefined): boolean {
  const sid = model?.successor;
  return typeof sid === 'string' && sid.trim().length > 0;
}

import type { CreativeMedia, CreativeTaskKind } from '@/composables/useCreativeAssistant';
import { DQ_STORAGE, consumeStringDraft, setItem } from '@/utils/storage';

export interface CopilotHandoff {
  media: CreativeMedia;
  task: CreativeTaskKind;
  assetId?: string;
  assetPreview?: string;
  prompt?: string;
}

export function setCopilotHandoff(handoff: CopilotHandoff): void {
  setItem(DQ_STORAGE.COPILOT_HANDOFF, JSON.stringify(handoff));
}

export function consumeCopilotHandoff(): CopilotHandoff | null {
  const raw = consumeStringDraft(DQ_STORAGE.COPILOT_HANDOFF);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as CopilotHandoff;
    if (!parsed?.media || !parsed?.task) return null;
    return parsed;
  } catch {
    return null;
  }
}

export function navigateToCopilot(
  router: { push: (loc: { name: string }) => void | Promise<unknown> },
  handoff: CopilotHandoff,
): void {
  setCopilotHandoff(handoff);
  void router.push({ name: 'assistant' });
}

export function assetIdFromGalleryPath(path: string): string | null {
  if (!path.startsWith('asset:')) return null;
  const id = path.slice('asset:'.length).trim();
  return id || null;
}

export function thumbnailUrlForAsset(assetId: string): string {
  return `/api/assets/${assetId}/thumbnail`;
}

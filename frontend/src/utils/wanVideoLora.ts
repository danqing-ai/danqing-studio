/** Wan 2.2 video Lightning LoRA helpers (I2V/T2V MoE high+low shards, 4-step distill). */

export const WAN_I2V_LIGHTNING_LORA_ID = 'wan2.2-i2v-lightning-lora';
export const WAN_T2V_LIGHTNING_LORA_ID = 'wan2.2-t2v-lightning-lora';

export type WanCompatibleLora = {
  id?: string;
  name?: string;
  base_model?: string;
  wan_lightning_distill?: boolean;
};

export function loraRegistryId(raw: string): string {
  return String(raw || '').split(':', 1)[0].trim();
}

export function isWanLightningLoraEntry(lora: WanCompatibleLora | null | undefined): boolean {
  if (!lora) return false;
  if (lora.wan_lightning_distill) return true;
  const id = loraRegistryId(String(lora.id || ''));
  return id === WAN_I2V_LIGHTNING_LORA_ID || id === WAN_T2V_LIGHTNING_LORA_ID;
}

export function findCompatibleLora(
  loras: WanCompatibleLora[],
  loraId: string,
): WanCompatibleLora | undefined {
  const key = String(loraId || '').trim();
  if (!key) return undefined;
  return loras.find((row) => String(row.id) === key);
}

/** Apply LightX2V Lightning schedule defaults when a Lightning LoRA is selected. */
export function applyWanLightningComposerParams(
  params: {
    steps?: number;
    guide_scale?: number;
    shift?: number;
  },
  lora: WanCompatibleLora | null | undefined,
): void {
  if (!isWanLightningLoraEntry(lora)) return;
  params.steps = 4;
  params.guide_scale = 0;
  params.shift = 5;
}

export function videoLoraOptionLabel(lora: WanCompatibleLora, myLoraTag?: string): string {
  const base = String(lora.name || lora.id || '');
  if ((lora as { source?: string }).source === 'user_trained' && myLoraTag) {
    return `${base} (${myLoraTag})`;
  }
  if (isWanLightningLoraEntry(lora)) {
    const id = loraRegistryId(String(lora.id || ''));
    if (id === WAN_I2V_LIGHTNING_LORA_ID) {
      return `${base} · I2V Lightning`;
    }
    if (id === WAN_T2V_LIGHTNING_LORA_ID) {
      return `${base} · T2V Lightning`;
    }
  }
  return base;
}

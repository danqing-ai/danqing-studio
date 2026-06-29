/** Parse / compose long-video scene beat strings from chapter analyze. */

export interface ParsedSceneBeat {
  shotSize: string;
  location: string;
  visual: string;
}

export interface ParsedCharacterLookBody {
  role: string;
  appearance: string;
  wardrobe: string;
}

const SHOT_SIZE_OPTIONS_ZH = ['远景', '全景', '中景', '近景', '特写'] as const;
const SHOT_SIZE_OPTIONS_EN = ['WS', 'FS', 'MS', 'MCU', 'CU'] as const;

export function shotSizeOptions(locale: string): readonly string[] {
  return locale.startsWith('zh') ? SHOT_SIZE_OPTIONS_ZH : SHOT_SIZE_OPTIONS_EN;
}

export function parseSceneBeat(beat: string): ParsedSceneBeat {
  const raw = (beat || '').trim();
  if (!raw) return { shotSize: '', location: '', visual: '' };

  const shotMatch = raw.match(/^【([^】]+)】([\s\S]*)$/);
  if (!shotMatch) {
    return { shotSize: '', location: '', visual: raw };
  }

  const shotSize = shotMatch[1].trim();
  const rest = shotMatch[2].trim();
  const commaIdx = rest.search(/[，,]/);
  if (commaIdx >= 0) {
    return {
      shotSize,
      location: rest.slice(0, commaIdx).trim(),
      visual: rest.slice(commaIdx + 1).trim(),
    };
  }
  return { shotSize, location: '', visual: rest };
}

export function composeSceneBeat(shotSize: string, location: string, visual: string): string {
  const shot = shotSize.trim();
  const loc = location.trim();
  const vis = visual.trim();
  if (!shot && !loc && !vis) return '';
  if (shot && loc && vis) return `【${shot}】${loc}，${vis}`;
  if (shot && vis) return `【${shot}】${vis}`;
  if (shot && loc) return `【${shot}】${loc}`;
  return vis || loc;
}

export function parseCharacterLookBody(body: string): ParsedCharacterLookBody {
  const raw = (body || '').trim();
  if (!raw) return { role: '', appearance: '', wardrobe: '' };

  const wardrobeMatch = raw.match(/(?:服装|Wardrobe)[:：]\s*([^|｜]+)/i);
  const wardrobe = wardrobeMatch?.[1]?.trim() ?? '';

  const zh = raw.match(/^定位[:：]\s*([^|｜]+?)\s*[|｜]\s*外貌[:：]\s*([\s\S]+)$/);
  if (zh) {
    const appearance = zh[2]
      .replace(/\s*[|｜]\s*(?:服装|Wardrobe)[:：][\s\S]*$/i, '')
      .trim();
    return { role: zh[1].trim(), appearance, wardrobe };
  }

  const en = raw.match(/^Role[:：]\s*([^|]+?)\s*\|\s*Appearance[:：]\s*([\s\S]+)$/i);
  if (en) {
    const appearance = en[2]
      .replace(/\s*\|\s*(?:Wardrobe)[:：][\s\S]*$/i, '')
      .trim();
    return { role: en[1].trim(), appearance, wardrobe };
  }

  return { role: '', appearance: raw, wardrobe };
}

export function splitSynopsisMood(text: string): { logline: string; mood: string } {
  const lines = (text || '')
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean);
  if (lines.length <= 1) {
    return { logline: lines[0] ?? '', mood: '' };
  }
  return { logline: lines[0], mood: lines.slice(1).join('\n') };
}

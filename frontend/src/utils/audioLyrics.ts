/** Registry-driven: vocal tracks need lyrics unless instrumental is on. */
export function isAudioLyricsRequired(
  modelConfig: { parameters?: Record<string, unknown> } | null | undefined,
  instrumental: boolean,
): boolean {
  if (instrumental) return false;
  const p = modelConfig?.parameters;
  if (!p) return false;
  if (p.requires_lyrics_for_vocal === true) return true;
  return p.supports_instrumental === false;
}

export function audioLyricsRequiredHintKey(
  modelConfig: { parameters?: Record<string, unknown> } | null | undefined,
): 'audio.lyricsRequiredAceHint' | 'audio.lyricsRequiredHint' {
  if (modelConfig?.parameters?.requires_lyrics_for_vocal === true) {
    return 'audio.lyricsRequiredAceHint';
  }
  return 'audio.lyricsRequiredHint';
}

/** Heuristic aligned with backend ``text_looks_chinese`` for vocal language. */
export function lyricsLookChinese(text: string): boolean {
  const cjk = (text.match(/[\u4e00-\u9fff]/g) || []).length;
  if (cjk < 2) return false;
  const latin = (text.match(/[A-Za-z]/g) || []).length;
  return cjk >= latin;
}

/** Prefer zh when lyrics are Chinese; avoids en/zh mismatch weakening ACE-Step vocals. */
export function resolveVocalLanguageForSubmit(lyrics: string, vocalLanguage: string): string {
  const explicit = (vocalLanguage || '').trim().toLowerCase();
  if (lyricsLookChinese(lyrics)) {
    if (!explicit || explicit === 'en' || explicit === 'english') return 'zh';
  }
  return vocalLanguage?.trim() || '';
}

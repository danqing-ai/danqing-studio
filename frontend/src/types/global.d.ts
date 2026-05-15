import 'vue';

declare module 'vue' {
  interface ComponentCustomProperties {
    $tt: (key: string, params?: Record<string, string | number>) => string;
    $mn: (model: { name?: string | { zh?: string; en?: string }; name_en?: string } | null, defaultName?: string) => string;
    $md: (model: { description?: string | { zh?: string; en?: string }; description_en?: string } | null, defaultDesc?: string) => string;
    $mvn: (modelKey: string, config: { name?: string | { zh?: string; en?: string }; name_en?: string } | null, versionConfig?: { name?: string | { zh?: string; en?: string } }) => string;
    $pn: (presetData: { name_en?: string } | null, chineseName?: string) => string;
  }
}
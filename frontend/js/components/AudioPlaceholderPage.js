/**
 * Audio creation placeholder page — Plan I1 (engine not yet connected, no legacy API surface)
 */
const AudioPlaceholderPage = {
    template: `
        <div class="create-page" style="padding: 40px 24px; max-width: 560px; margin: 0 auto;">
            <div style="text-align: center; margin-bottom: 32px;">
                <el-icon :size="56" color="var(--primary)"><Microphone /></el-icon>
                <h2 style="margin-top: 20px; color: var(--text-primary); font-weight: 600;">{{ $t('audio.title') }}</h2>
                <p style="margin-top: 12px; color: var(--text-muted); line-height: 1.6;">{{ $t('audio.desc') }}</p>
            </div>
            <div class="card" style="padding: 20px;">
                <div style="font-weight: 600; margin-bottom: 12px; color: var(--text-primary);">{{ $t('audio.plannedTitle') }}</div>
                <ul style="margin: 0; padding-left: 20px; color: var(--text-muted); line-height: 1.8;">
                    <li>{{ $t('audio.planned1') }}</li>
                    <li>{{ $t('audio.planned2') }}</li>
                    <li>{{ $t('audio.planned3') }}</li>
                </ul>
            </div>
            <div class="card" style="padding: 20px; margin-top: 16px;">
                <div style="font-weight: 600; margin-bottom: 12px; color: var(--text-primary);">{{ $t('audio.contractTitle') }}</div>
                <ul style="margin: 0; padding-left: 20px; color: var(--text-muted); line-height: 1.8;">
                    <li>{{ $t('audio.contractGenerate') }}</li>
                    <li>{{ $t('audio.contractEdit') }}</li>
                    <li>{{ $t('audio.contractDub') }}</li>
                </ul>
            </div>
            <p style="margin-top: 24px; font-size: 13px; color: var(--text-muted); text-align: center;">{{ $t('audio.apiNote') }}</p>
        </div>
    `,
    setup() {
        const { Microphone } = ElementPlusIconsVue;
        return { Microphone };
    },
};

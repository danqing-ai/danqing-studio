/**
 * TopNav — 顶部导航内容组件（不含 el-header 外壳，由父级 index.html 提供）
 * Props:  activePage (String), queueCount (Number)
 * Emits:  navigate(page), open-queue
 */
const TopNav = {
    name: 'TopNav',
    props: {
        activePage: { type: String, default: 'image_create' },
        queueCount: { type: Number, default: 0 },
    },
    emits: ['navigate', 'open-queue'],
    template: `
        <div style="display: flex; align-items: center; justify-content: space-between; width: 100%; height: 60px;">
            <div class="header-brand">
                <el-icon size="28" color="#e94560"><magic-stick /></el-icon>
                <span class="brand-title">DanQing Studio</span>
                <span class="brand-subtitle">v3</span>
            </div>

            <el-menu
                ref="navMenuRef"
                :default-active="activePage"
                mode="horizontal"
                class="nav-menu"
                @select="onNavSelect"
                background-color="transparent"
                text-color="#a0a3bd"
                active-text-color="#e94560"
            >
                <el-menu-item index="image_create">
                    <el-icon><brush /></el-icon>
                    <span>{{ $t('nav.image_create') }}</span>
                </el-menu-item>
                <el-menu-item index="video_create">
                    <el-icon><video-camera /></el-icon>
                    <span>{{ $t('nav.video_create') }}</span>
                </el-menu-item>
                <el-menu-item index="audio_create">
                    <el-icon><microphone /></el-icon>
                    <span>{{ $t('nav.audio_create') }}</span>
                </el-menu-item>
                <el-menu-item index="gallery">
                    <el-icon><picture-filled /></el-icon>
                    <span>{{ $t('nav.gallery') }}</span>
                </el-menu-item>
                <el-menu-item index="models">
                    <el-icon><download /></el-icon>
                    <span>{{ $t('nav.models') }}</span>
                </el-menu-item>
                <el-menu-item index="settings">
                    <el-icon><setting /></el-icon>
                    <span>{{ $t('nav.settings') }}</span>
                </el-menu-item>
            </el-menu>

            <div class="header-actions">
                <el-badge :value="queueCount" :hidden="queueCount === 0" class="queue-badge">
                    <el-button
                        @click="openQueue"
                        :title="$t('studio.taskQueue')"
                    >
                        <el-icon><document-copy /></el-icon>
                        <span>{{ $t('studio.taskQueue') }}</span>
                    </el-button>
                </el-badge>
            </div>
        </div>
    `,
    setup(props, { emit }) {
        const navMenuRef = Vue.ref(null);

        function onNavSelect(index) {
            emit('navigate', index);
        }

        function openQueue() {
            emit('open-queue');
        }

        Vue.watch(function () { return props.activePage; }, function (newVal) {
            Vue.nextTick(function () {
                if (navMenuRef.value) {
                    navMenuRef.value.activeIndex = newVal;
                }
            });
        });

        return { navMenuRef, onNavSelect, openQueue };
    },
};

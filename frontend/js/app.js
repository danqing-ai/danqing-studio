/**
 * Vue application entry point
 */

const { createApp, ref, reactive, onMounted, onBeforeUnmount, provide, watch, nextTick, computed } = Vue;

// Check if VueI18n is loaded
if (typeof VueI18n === 'undefined') {
    console.error('VueI18n not loaded, check CDN link');
}
if (typeof messages === 'undefined') {
    console.error('messages not defined, check if i18n.js is loaded');
}

// Create i18n instance
let i18n;
try {
    i18n = VueI18n.createI18n({
        locale: 'zh',
        fallbackLocale: 'en',
        messages: messages || {},
        legacy: false,
        missing: (locale, key) => {
            console.warn(`Missing translation: ${key}`);
            return key;
        }
    });
} catch (e) {
    console.error('i18n initialization failed:', e);
    i18n = {
        install: (app) => {
            app.config.globalProperties.$t = (key) => key;
        },
        global: {
            locale: { value: 'zh' },
            t: (key) => key
        }
    };
}

window.$tt = (key, params = {}) => {
    try {
        const result = i18n.global.t(key, params);
        return result || key;
    } catch (e) {
        return key;
    }
};

window.$mn = (model, defaultName) => {
    if (!model) return defaultName || '';
    const locale = i18n.global.locale.value;
    const n = model.name;
    if (n && typeof n === 'object') {
        return locale === 'en' ? (n.en || n.zh || defaultName || '') : (n.zh || n.en || defaultName || '');
    }
    if (locale === 'en' && model.name_en) return model.name_en;
    return n || defaultName || '';
};

window.$md = (model, defaultDesc) => {
    if (!model) return defaultDesc || '';
    const locale = i18n.global.locale.value;
    const d = model.description;
    if (d && typeof d === 'object') {
        return locale === 'en' ? (d.en || d.zh || defaultDesc || '') : (d.zh || d.en || defaultDesc || '');
    }
    if (locale === 'en' && model.description_en) return model.description_en;
    return d || defaultDesc || '';
};

/** Registry model + version row display (config.name may be {zh,en}, version.name may also be an object) */
window.$mvn = (modelKey, config, versionConfig) => {
    const base = window.$mn ? window.$mn(config, modelKey) : modelKey;
    const vn = versionConfig && versionConfig.name;
    if (vn == null || vn === '') return base;
    let suffix = '';
    if (typeof vn === 'object' && vn !== null && ('zh' in vn || 'en' in vn)) {
        const locale = i18n.global.locale.value;
        suffix = locale === 'en' ? (vn.en || vn.zh || '') : (vn.zh || vn.en || '');
    } else {
        suffix = String(vn);
    }
    return suffix ? `${base} - ${suffix}` : base;
};

/** Sync with settings page `theme`: add root class for light, remove for dark */
window.DQApplyTheme = (theme) => {
    const el = document.documentElement;
    if (theme === 'light') {
        el.classList.add('dq-theme-light');
    } else {
        el.classList.remove('dq-theme-light');
    }
};

window.$pn = (presetData, chineseName) => {
    const locale = i18n.global.locale.value;
    if (locale === 'en' && presetData && presetData.name_en) return presetData.name_en;
    return chineseName || presetData?.name_en || '';
};

const app = createApp({
    setup() {
        // ---- Router integration ----
        const router = window.DQRouter;
        // Use a local ref to guarantee Vue template reactivity; sync with DQRouter bidirectionally
        const initialPage = router ? router.currentPage.value : 'image_create';
        const activePage = ref(initialPage);

        if (router) {
            // router → local (hash change / back-forward)
            watch(router.currentPage, function (newVal) {
                if (activePage.value !== newVal) {
                    activePage.value = newVal;
                }
            });
            // local → router (nav click / programmatic)
            watch(activePage, function (newVal) {
                if (router.currentPage.value !== newVal) {
                    router.navigate(newVal);
                }
            });
        }

        const currentLang = ref('zh');
        const showGlobalQueueDrawer = ref(false);

        const globalTaskQueue = computed(() => {
            const MQ = window.DQMediaQueue;
            if (MQ && typeof MQ.snapshotFullQueue === 'function') {
                return MQ.snapshotFullQueue(window.TasksStore);
            }
            return { running: [], queued: [] };
        });

        const globalQueueCount = computed(
            () => globalTaskQueue.value.running.length + globalTaskQueue.value.queued.length
        );

        const systemInfo = reactive({
            env_ready: false,
            platform: '',
            architecture: '',
            memory_gb: 0,
            mlx_memory_limit: 120,
        });

        // ---- Navigation ----

        function handleNavSelect(index) {
            activePage.value = index;
        }

        // ---- Task operations ----

        const cancelGlobalTask = async (taskId) => {
            const TS = window.TasksStore;
            try {
                await api.gen.cancelMediaTask(taskId);
                if (TS && typeof TS.pollQueueOnce === 'function') {
                    await TS.pollQueueOnce();
                }
                if (typeof ElementPlus !== 'undefined' && ElementPlus.ElMessage) {
                    ElementPlus.ElMessage.success($tt('studio.cancelled'));
                }
            } catch (e) {
                console.error('cancelGlobalTask', e);
                if (typeof ElementPlus !== 'undefined' && ElementPlus.ElMessage) {
                    ElementPlus.ElMessage.error($tt('studio.error', { msg: e.message || String(e) }));
                }
            }
        };

        const setQueuedPriority = async (taskId, priority) => {
            const TS = window.TasksStore;
            try {
                await api.gen.patchMediaTaskPriority(taskId, { priority });
                if (TS && typeof TS.pollQueueOnce === 'function') {
                    await TS.pollQueueOnce();
                }
                if (typeof ElementPlus !== 'undefined' && ElementPlus.ElMessage) {
                    ElementPlus.ElMessage.success($tt('studio.priorityUpdated'));
                }
            } catch (e) {
                console.error('setQueuedPriority', e);
                const msg =
                    (e.response && e.response.data && e.response.data.detail) ||
                    e.message ||
                    String(e);
                if (typeof ElementPlus !== 'undefined' && ElementPlus.ElMessage) {
                    ElementPlus.ElMessage.error($tt('studio.error', { msg }));
                }
            }
        };

        // ---- System info ----

        const loadSystemInfo = async () => {
            try {
                const info = await api.settings.getSystemInfo();
                Object.assign(systemInfo, info);
            } catch (e) {
                console.error('Failed to load system info:', e);
            }
        };

        // ---- Lifecycle ----

        onMounted(async () => {
            // Load language setting
            const SK = window.DQ_STORAGE || {};
            const savedLang = SK.LANG ? localStorage.getItem(SK.LANG) : null;
            if (savedLang) {
                currentLang.value = savedLang;
                i18n.global.locale.value = savedLang;
                document.documentElement.lang = savedLang;
            }

            // Load system info
            loadSystemInfo();

            try {
                const st = await api.settings.getSettings();
                if (st && st.theme && typeof window.DQApplyTheme === 'function') {
                    window.DQApplyTheme(st.theme);
                }
            } catch (e) {
                console.warn('Theme bootstrap skipped:', e);
            }

            // Periodically refresh system info
            setInterval(loadSystemInfo, 30000);

            const TS = window.TasksStore;
            if (TS && typeof TS.ensureQueuePoller === 'function') {
                TS.ensureQueuePoller();
            }

            // Listen for global navigation events (e.g., from creation page to settings page)
            window.addEventListener('navigate', (e) => {
                if (e.detail) {
                    handleNavSelect(e.detail);
                }
            });

            window.addEventListener('open-global-task-queue', onOpenGlobalTaskQueue);
        });

        function onOpenGlobalTaskQueue() {
            showGlobalQueueDrawer.value = true;
        }

        onBeforeUnmount(() => {
            window.removeEventListener('open-global-task-queue', onOpenGlobalTaskQueue);
            const TS = window.TasksStore;
            if (TS && typeof TS.releaseQueuePoller === 'function') {
                TS.releaseQueuePoller();
            }
        });

        // Provide global state
        provide('systemInfo', systemInfo);

        function openTaskQueue() {
            showGlobalQueueDrawer.value = true;
        }

        function queueTruncate(text, length) {
            if (!text) return '';
            return text.length > length ? text.substring(0, length) + '...' : text;
        }

        function taskKindLabel(kind) {
            if (!kind) return '';
            var key = 'taskKind.' + String(kind).replace(/\./g, '_');
            try {
                var r = i18n.global.t(key);
                return r && r !== key ? r : kind;
            } catch (_) {
                return kind;
            }
        }

        return {
            activePage,
            currentLang,
            systemInfo,
            handleNavSelect,
            openTaskQueue,
            showGlobalQueueDrawer,
            globalQueueCount,
            globalTaskQueue,
            cancelGlobalTask,
            setQueuedPriority,
            queueTruncate,
            taskKindLabel,
        };
    }
});

// Register shell component
app.component('TopNav', TopNav);

// Register all page components
app.component('AdapterPicker', AdapterPicker);
app.component('AssetPicker', AssetPicker);
app.component('RegistryParamsForm', RegistryParamsForm);
app.component('ImageEditor', ImageEditor);
app.component('ImageCreatePage', ImageCreatePage);
app.component('VideoCreatePage', VideoCreatePage);
app.component('GalleryPage', GalleryPage);
app.component('ModelsPage', ModelsPage);
app.component('SettingsPage', SettingsPage);
app.component('AudioCreatePage', AudioCreatePage);

// Register Element Plus icons (safe loading)
try {
    if (typeof ElementPlusIconsVue !== 'undefined') {
        for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
            app.component(key, component);
        }
    }
} catch (e) {
    console.warn('Icon registration failed (does not affect core functionality):', e);
}

app.mixin({
    methods: {
        $mn: window.$mn,
        $md: window.$md,
        $mvn: window.$mvn,
        $pn: window.$pn,
        $tt: window.$tt,
    },
});
app.use(ElementPlus);
app.use(i18n);
app.mount('#app');

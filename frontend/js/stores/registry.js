/**
 * 只读注册表缓存 — Plan C1（GET /api/registry）
 * CDN 无打包：挂载 window.RegistryStore
 */
(function (w) {
    const API_BASE = '';
    const { ref } = Vue;
    const registry = ref(null);
    const loading = ref(false);
    let lastLoad = 0;

    async function load(force) {
        const now = Date.now();
        if (!force && registry.value && now - lastLoad < 60_000) {
            return registry.value;
        }
        loading.value = true;
        try {
            const data =
                typeof w.api !== 'undefined' && w.api.registry && typeof w.api.registry.getFull === 'function'
                    ? await w.api.registry.getFull()
                    : (await axios.get(`${API_BASE}/api/registry`)).data;
            registry.value = data;
            lastLoad = now;
            return registry.value;
        } finally {
            loading.value = false;
        }
    }

    w.RegistryStore = { registry, loading, load };
})(window);

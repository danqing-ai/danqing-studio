import { ref } from 'vue';
import { defineStore } from 'pinia';
import { api } from '@/utils/api';
import type { RegistryData } from '@/types';

export const useRegistryStore = defineStore('registry', () => {
  const registry = ref<RegistryData | null>(null);
  const loading = ref(false);
  let lastLoad = 0;

  async function load(force = false): Promise<RegistryData | null> {
    const now = Date.now();
    if (!force && registry.value && now - lastLoad < 60_000) {
      return registry.value;
    }
    loading.value = true;
    try {
      const data = await api.registry.getFull();
      registry.value = data;
      lastLoad = now;
      return registry.value;
    } finally {
      loading.value = false;
    }
  }

  return { registry, loading, load };
});
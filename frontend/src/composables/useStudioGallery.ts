import { ref, computed, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { api } from '@/utils/api';
import { confirm, toast } from '@/utils/feedback';
import { $tt } from '@/utils/i18n';
import type { GalleryItem } from '@/types';

const GALLERY_PAGE_SIZE = 40;

export function useStudioGallery(mediaKind: 'image' | 'video' | 'audio') {
  const { t: $t } = useI18n();

  const galleryItems = ref<GalleryItem[]>([]);
  const galleryLoading = ref(false);
  const galleryHasMore = ref(true);
  const galleryOffset = ref(0);

  const filterTime = ref('all');
  const filterModels = ref<string[]>([]);
  const selectionMode = ref(false);
  const selectedItems = ref<GalleryItem[]>([]);

  const timeOptions = computed(() => [
    { value: 'all', label: $t('gallery.dateAll') },
    { value: 'today', label: $t('gallery.dateToday') },
    { value: 'week', label: $t('gallery.dateWeek') },
    { value: 'month', label: $t('gallery.dateMonth') },
  ]);

  const allModelOptions = computed(() => {
    const set = new Set<string>();
    galleryItems.value.forEach((it) => {
      if (it.model) set.add(it.model);
    });
    return Array.from(set).sort();
  });

  const hasActiveFilters = computed(() => {
    return filterTime.value !== 'all' || filterModels.value.length > 0;
  });

  const selectedPaths = computed(() => new Set(selectedItems.value.map((it) => it.path)));

  const allLoadedSelected = computed(() => {
    return galleryItems.value.length > 0 && galleryItems.value.every((it) => selectedPaths.value.has(it.path));
  });

  function buildGalleryFilterParams() {
    const params: Record<string, unknown> = { kind: mediaKind };
    if (filterTime.value !== 'all') {
      const now = new Date();
      let date: Date | undefined;
      switch (filterTime.value) {
        case 'today':
          date = new Date(now.getFullYear(), now.getMonth(), now.getDate());
          break;
        case 'week':
          date = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
          break;
        case 'month':
          date = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
          break;
      }
      if (date) {
        params.created_after = date.toISOString();
      }
    }
    if (filterModels.value.length > 0) {
      params.model = filterModels.value[0];
    }
    return params;
  }

  function clearSelection() {
    selectedItems.value = [];
    selectionMode.value = false;
  }

  async function loadGallery(reset = false) {
    if (galleryLoading.value) return;
    if (reset) {
      galleryItems.value = [];
      galleryOffset.value = 0;
      galleryHasMore.value = true;
      clearSelection();
    }
    if (!galleryHasMore.value && !reset) return;

    galleryLoading.value = true;
    try {
      const options = buildGalleryFilterParams();
      const rows = await api.gallery.listImages(GALLERY_PAGE_SIZE, galleryOffset.value, options);
      if (reset) {
        galleryItems.value = rows || [];
      } else {
        galleryItems.value.push(...(rows || []));
      }
      if ((rows || []).length < GALLERY_PAGE_SIZE) {
        galleryHasMore.value = false;
      } else {
        galleryOffset.value += GALLERY_PAGE_SIZE;
      }
    } catch (e) {
      console.error('Failed to load gallery:', e);
      toast.error($tt('gallery.loadFailed'));
    } finally {
      galleryLoading.value = false;
    }
  }

  function refreshGallery() {
    loadGallery(true);
  }

  function onCanvasScroll(e: Event) {
    const el = e.target as HTMLElement;
    const bottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (bottom < 200 && !galleryLoading.value && galleryHasMore.value) {
      loadGallery(false);
    }
  }

  function resetGalleryFilters() {
    filterTime.value = 'all';
    filterModels.value = [];
  }

  function isSelected(item: GalleryItem) {
    return selectedPaths.value.has(item.path);
  }

  function toggleSelect(item: GalleryItem) {
    const idx = selectedItems.value.findIndex((it) => it.path === item.path);
    if (idx >= 0) {
      selectedItems.value.splice(idx, 1);
    } else {
      selectedItems.value.push(item);
    }
  }

  function toggleSelectionMode() {
    if (selectionMode.value) {
      clearSelection();
      return;
    }
    selectionMode.value = true;
  }

  function selectAllLoaded() {
    if (allLoadedSelected.value) {
      selectedItems.value = [];
      return;
    }
    selectedItems.value = [...galleryItems.value];
  }

  function handleCardClick(item: GalleryItem, event?: MouseEvent) {
    if (selectionMode.value) {
      toggleSelect(item);
      return;
    }
    if (event && (event.ctrlKey || event.metaKey || event.shiftKey)) {
      selectionMode.value = true;
      toggleSelect(item);
    }
  }

  function removeItemsByPath(paths: string[]) {
    const pathSet = new Set(paths);
    galleryItems.value = galleryItems.value.filter((it) => !pathSet.has(it.path));
    selectedItems.value = selectedItems.value.filter((it) => !pathSet.has(it.path));
    if (selectedItems.value.length === 0) {
      selectionMode.value = false;
    }
  }

  async function deleteItem(item: GalleryItem) {
    try {
      await confirm($tt('gallery.confirmDelete'), $tt('gallery.confirmDeleteTitle'), { type: 'warning' });
      await api.gallery.deleteImage(item.path);
      toast.success($tt('gallery.deleted'));
      removeItemsByPath([item.path]);
    } catch (e) {
      if (e !== 'cancel') {
        console.error('Delete failed:', e);
        toast.error($tt('gallery.batchDeleteFailed'));
      }
    }
  }

  async function batchDeleteSelected() {
    if (selectedItems.value.length === 0) return;
    try {
      await confirm(
        $tt('gallery.batchDeleteConfirm', { count: selectedItems.value.length }),
        $tt('gallery.confirmDeleteTitle'),
        { type: 'warning' }
      );
      const paths = selectedItems.value.map((it) => it.path);
      await api.gallery.batchDeleteImages(paths);
      toast.success($tt('gallery.batchDeleted', { count: paths.length }));
      removeItemsByPath(paths);
    } catch (e) {
      if (e !== 'cancel') {
        console.error('Batch delete failed:', e);
        toast.error($tt('gallery.batchDeleteFailed'));
      }
    }
  }

  watch([filterTime, filterModels], () => {
    loadGallery(true);
  });

  return {
    galleryItems,
    galleryLoading,
    galleryHasMore,
    filterTime,
    filterModels,
    selectionMode,
    selectedItems,
    selectedPaths,
    allLoadedSelected,
    timeOptions,
    allModelOptions,
    hasActiveFilters,
    loadGallery,
    refreshGallery,
    onCanvasScroll,
    resetGalleryFilters,
    isSelected,
    toggleSelect,
    toggleSelectionMode,
    selectAllLoaded,
    handleCardClick,
    deleteItem,
    batchDeleteSelected,
    clearSelection,
  };
}

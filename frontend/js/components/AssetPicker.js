/**
 * Plan C4：从上传、最近图库条、资产库选择 `asset:{id}`，统一走 /api/assets。
 */
const AssetPicker = {
    name: 'AssetPicker',
    props: {
        recentGallery: { type: Array, default: () => [] },
        /** 过滤最近条与资产库列表：`image` | `video` */
        acceptKind: { type: String, default: 'image' },
    },
    emits: ['pick'],
    data() {
        return {
            libOpen: false,
            libLoading: false,
            libItems: [],
        };
    },
    computed: {
        fileAccept() {
            return this.acceptKind === 'video' ? 'video/*' : 'image/*';
        },
        filteredRecent() {
            const list = Array.isArray(this.recentGallery) ? this.recentGallery : [];
            const vidExt = ['mp4', 'mov', 'avi', 'mkv', 'webm'];
            return list
                .filter((v) => {
                    const kind = v.metadata && v.metadata.asset_kind;
                    const ext = (v.name || '').split('.').pop()?.toLowerCase() || '';
                    if (this.acceptKind === 'video') {
                        if (kind === 'video') return true;
                        return vidExt.includes(ext);
                    }
                    if (kind === 'video') return false;
                    return !vidExt.includes(ext);
                })
                .slice(0, 12);
        },
    },
    methods: {
        thumbForGalleryItem(it) {
            if (it.thumbnail) return it.thumbnail;
            if (typeof it.path === 'string' && it.path.startsWith('asset:')) {
                return api.gallery.getImageUrl(it.path);
            }
            return '';
        },
        thumbForAssetRow(row) {
            if (row.thumbnail_url) return row.thumbnail_url;
            return api.gallery.getImageUrl('asset:' + row.id);
        },
        pickRecent(it) {
            if (!it.path || typeof it.path !== 'string' || !it.path.startsWith('asset:')) return;
            this.$emit('pick', {
                path: it.path,
                previewUrl: api.gallery.getImageUrl(it.path),
            });
        },
        pickLib(row) {
            const path = 'asset:' + row.id;
            this.$emit('pick', {
                path,
                previewUrl: api.gallery.getImageUrl(path),
            });
            this.libOpen = false;
        },
        async openLibrary() {
            this.libOpen = true;
            this.libLoading = true;
            this.libItems = [];
            try {
                const kind = this.acceptKind === 'video' ? 'video' : 'image';
                const data = await api.gen.listAssets(kind, 60, 0);
                this.libItems = data.items || [];
            } catch (err) {
                ElementPlus.ElMessage.error(
                    window.$tt('studio.uploadFailed', { msg: err.message || String(err) })
                );
            } finally {
                this.libLoading = false;
            }
        },
        async onPickFile(ev) {
            const file = ev.target.files && ev.target.files[0];
            if (!file) return;
            try {
                const r = await api.gen.uploadAsset(file);
                const path = 'asset:' + r.id;
                if (this.acceptKind === 'image' && r.kind && r.kind !== 'image') {
                    ElementPlus.ElMessage.warning(window.$tt('assetPicker.needImage'));
                    ev.target.value = '';
                    return;
                }
                this.$emit('pick', {
                    path,
                    previewUrl: api.gallery.getImageUrl(path),
                });
            } catch (err) {
                ElementPlus.ElMessage.error(
                    window.$tt('studio.uploadFailed', { msg: err.message || String(err) })
                );
            }
            ev.target.value = '';
        },
    },
    template: `
        <div class="asset-picker">
            <div class="asset-picker__actions">
                <input
                    ref="fileInput"
                    type="file"
                    :accept="fileAccept"
                    style="display: none;"
                    @change="onPickFile"
                />
                <el-button size="small" @click="$refs.fileInput.click()">{{ $t('assetPicker.upload') }}</el-button>
                <el-button size="small" @click="openLibrary">{{ $t('assetPicker.library') }}</el-button>
            </div>
            <div v-if="filteredRecent.length" class="asset-picker__recent">
                <span class="asset-picker__recent-label">{{ $t('assetPicker.recent') }}</span>
                <div class="asset-picker__recent-grid">
                    <div
                        v-for="it in filteredRecent"
                        :key="it.path"
                        class="asset-picker__thumb"
                        @click="pickRecent(it)"
                    >
                        <img :src="thumbForGalleryItem(it)" alt="" />
                    </div>
                </div>
            </div>
            <el-dialog v-model="libOpen" :title="$t('assetPicker.dialogTitle')" width="580px">
                <el-empty v-if="!libLoading && (!libItems || libItems.length === 0)" :description="$t('assetPicker.emptyLibrary')" />
                <div v-loading="libLoading" style="display: flex; flex-wrap: wrap; gap: 8px; max-height: 52vh; overflow-y: auto;">
                    <div
                        v-for="row in libItems"
                        :key="row.id"
                        style="width: 96px; height: 96px; border-radius: 8px; overflow: hidden; cursor: pointer; border: 1px solid var(--border-color);"
                        @click="pickLib(row)"
                    >
                        <img :src="thumbForAssetRow(row)" alt="" style="width: 100%; height: 100%; object-fit: cover;" />
                    </div>
                </div>
            </el-dialog>
        </div>
    `,
};

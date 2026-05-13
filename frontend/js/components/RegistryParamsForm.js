/**
 * Registry-driven image advanced parameters form (steps / CFG / scheduler / resolution / denoising strength / seed, etc.).
 * LoRA, ControlNet depend on runtime lists passed as props by the parent component.
 */
const RegistryParamsForm = {
    name: 'RegistryParamsForm',
    props: {
        modelConfig: { type: Object, default: null },
        /** Mutable object shared with ImageCreatePage.params */
        params: { type: Object, required: true },
        /** Whether each parameter field is visible; defaults to true */
        paramVisibility: { type: Object, default: () => ({}) },
        /** LoRA list compatible with current model; pass null to hide LoRA row (even if model declares lora_support) */
        loras: { type: Array, default: null },
        /** ControlNet list compatible with current model */
        controlnets: { type: Array, default: null },
        controlImageSrc: { type: String, default: '' },
        /** ControlNet reference strip: shared with creation page recent gallery */
        controlRecentGallery: { type: Array, default: () => [] },
    },
    template: `
        <el-form label-position="top" size="small" style="padding-top: 12px;" v-if="modelConfig">
            <!-- Resolution width + height -->
            <el-form-item v-if="resPair && visible('width')" :label="$t('studio.resolution')">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <el-select v-model="params.width" style="width: 120px;">
                        <el-option
                            v-for="w in resPair.width.options"
                            :key="w"
                            :label="String(w)"
                            :value="w"
                        />
                    </el-select>
                    <span style="color: var(--text-muted);">x</span>
                    <el-select v-model="params.height" style="width: 120px;">
                        <el-option
                            v-for="h in resPair.height.options"
                            :key="h"
                            :label="String(h)"
                            :value="h"
                        />
                    </el-select>
                </div>
            </el-form-item>

            <template v-for="key in scalarKeys" :key="key">
                <el-form-item v-if="visible(key) && specOf(key)" :label="paramLabel(key, specOf(key))">
                    <!-- int / float slider -->
                    <template v-if="specOf(key).type === 'int' || specOf(key).type === 'float'">
                        <div class="param-control-row">
                            <div class="param-slider">
                                <el-slider
                                    v-model="params[key]"
                                    :min="specOf(key).min"
                                    :max="specOf(key).max"
                                    :step="numStep(key, specOf(key))"
                                />
                            </div>
                            <el-input-number
                                v-model="params[key]"
                                :min="specOf(key).min"
                                :max="specOf(key).max"
                                :step="numStep(key, specOf(key))"
                                class="param-input-number"
                            />
                        </div>
                    </template>
                    <!-- enum -->
                    <el-select v-else-if="specOf(key).type === 'enum'" v-model="params[key]" style="width: 100%;">
                        <el-option
                            v-for="opt in specOf(key).options"
                            :key="String(opt)"
                            :label="String(opt)"
                            :value="opt"
                        />
                    </el-select>
                </el-form-item>
            </template>

            <!-- LoRA / Adapter — Plan F3 AdapterPicker -->
            <adapter-picker
                v-if="showLoraBlock"
                :items="adapterItems"
                :adapter-id="params.lora"
                @update:adapter-id="params.lora = $event"
                :weight="params.lora_scale"
                @update:weight="params.lora_scale = $event"
                :weight-spec="loraScaleSpec"
            />

            <!-- ControlNet -->
            <el-form-item v-if="showControlnetBlock" :label="$t('studio.controlNet')">
                <div style="display: flex; flex-direction: column; gap: 8px; width: 100%;">
                    <el-select v-model="params.controlnet" clearable :placeholder="$t('studio.noControlNet')">
                        <el-option
                            v-for="net in controlnets"
                            :key="net.key"
                            :label="$mn ? $mn(net) : net.name"
                            :value="net.key"
                            :disabled="!net.ready"
                        />
                    </el-select>
                    <div v-if="params.controlnet" style="display: flex; flex-direction: column; gap: 8px;">
                        <div v-if="controlImageSrc" style="display: flex; align-items: flex-start; gap: 10px;">
                            <div class="ref-image-thumb" style="width: 80px; height: 80px; flex-shrink: 0;">
                                <img :src="controlImageSrc" alt="control" style="width: 100%; height: 100%; object-fit: cover; border-radius: 6px;" />
                                <el-button size="small" circle type="danger" @click="$emit('remove-control-image')" style="position: absolute; top: -6px; right: -6px;">
                                    <el-icon><Delete /></el-icon>
                                </el-button>
                            </div>
                            <span style="font-size: 12px; color: var(--text-muted); white-space: pre-line;">{{ $t('studio.uploadControlImage') }}</span>
                        </div>
                        <div v-else>
                            <asset-picker
                                accept-kind="image"
                                :recent-gallery="controlRecentGallery"
                                @pick="$emit('control-asset-pick', $event)"
                            />
                            <span style="font-size: 12px; color: var(--text-muted); white-space: pre-line; display: block; margin-top: 6px;">{{ $t('studio.uploadControlImage') }}</span>
                        </div>
                        <div class="param-control-row" v-if="cnStrengthSpec">
                            <div class="param-slider">
                                <el-slider
                                    v-model="params.controlnet_strength"
                                    :min="cnStrengthSpec.min"
                                    :max="cnStrengthSpec.max"
                                    :step="cnStrengthSpec.step || 0.05"
                                />
                            </div>
                            <el-input-number
                                v-model="params.controlnet_strength"
                                :min="cnStrengthSpec.min"
                                :max="cnStrengthSpec.max"
                                :step="cnStrengthSpec.step || 0.05"
                                style="width: 100px;"
                            />
                        </div>
                    </div>
                </div>
            </el-form-item>

            <!-- Seed -->
            <el-form-item v-if="seedSupport && visible('seed')" :label="$t('studio.seed')">
                <div style="display: flex; gap: 8px;">
                    <el-input v-model="params.seed" :placeholder="$t('studio.seedPlaceholder')" style="flex: 1;" />
                    <el-button @click="params.seed = String(Math.floor(Math.random() * 1_000_000))">
                        <el-icon><refresh /></el-icon>
                    </el-button>
                </div>
            </el-form-item>

            <el-form-item>
                <el-button text type="primary" @click="$emit('restore-defaults')" size="small">
                    <el-icon><refresh /></el-icon>
                    {{ $t('studio.restoreDefaults') }}
                </el-button>
            </el-form-item>
        </el-form>
    `,
    emits: ['control-asset-pick', 'remove-control-image', 'restore-defaults'],
    computed: {
        normalized() {
            const R = window.RegistryParamSchema;
            if (!R || !this.modelConfig || !this.modelConfig.parameters) return {};
            return R.normalizeParamsDef(this.modelConfig.parameters);
        },
        resPair() {
            const R = window.RegistryParamSchema;
            return R ? R.resolutionPair(this.normalized) : null;
        },
        scalarKeys() {
            const R = window.RegistryParamSchema;
            return R ? R.scalarKeysForForm(this.normalized) : [];
        },
        seedSupport() {
            return !!(this.modelConfig && this.modelConfig.parameters && this.modelConfig.parameters.seed_support);
        },
        showLoraBlock() {
            const p = this.modelConfig && this.modelConfig.parameters;
            if (!p || !p.lora_support) return false;
            return Array.isArray(this.loras);
        },
        adapterItems() {
            if (!Array.isArray(this.loras)) return [];
            return this.loras.map((l) => ({
                kind: 'lora',
                id: l.id,
                name: l.name,
            }));
        },
        showControlnetBlock() {
            return Array.isArray(this.controlnets) && this.controlnets.length > 0;
        },
        loraScaleSpec() {
            const s = this.normalized.lora_scale;
            if (s && (s.type === 'int' || s.type === 'float')) {
                return {
                    min: s.min ?? 0,
                    max: s.max ?? 2,
                    step: s.step ?? 0.1,
                };
            }
            return { min: 0, max: 2, step: 0.1 };
        },
        cnStrengthSpec() {
            const s = this.normalized.controlnet_strength;
            if (s && (s.type === 'int' || s.type === 'float')) {
                return s;
            }
            return { min: 0, max: 1, step: 0.05, default: 0.8 };
        },
    },
    methods: {
        visible(key) {
            if (this.paramVisibility && Object.prototype.hasOwnProperty.call(this.paramVisibility, key)) {
                return !!this.paramVisibility[key];
            }
            return true;
        },
        specOf(key) {
            return this.normalized[key];
        },
        numStep(key, spec) {
            if (typeof spec.step === 'number') return spec.step;
            return spec.type === 'int' ? 1 : 0.1;
        },
        paramLabel(key, spec) {
            const map = {
                steps: 'create.stepsLabel',
                guidance: 'create.guidanceLabel',
                scheduler: 'create.schedulerLabel',
                strength: 'create.strengthLabel',
                controlnet_strength: 'create.controlNetStrengthLabel',
                redux_strength: 'create.reduxStrengthLabel',
            };
            const i18nKey = map[key];
            if (i18nKey) {
                try {
                    return this.$t(i18nKey);
                } catch (e) {
                    /* fall through */
                }
            }
            if (spec && spec.label) return spec.label;
            return key;
        },
    },
};

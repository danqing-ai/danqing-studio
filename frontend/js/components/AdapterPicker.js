/**
 * Plan F3: LoRA / future registry adapter — two-way binding with params.lora, params.lora_scale
 */
const AdapterPicker = {
    name: 'AdapterPicker',
    props: {
        items: { type: Array, default: () => [] },
        adapterId: { type: String, default: '' },
        weight: { type: Number, default: 0.8 },
        weightSpec: {
            type: Object,
            default: () => ({ min: 0, max: 2, step: 0.1 }),
        },
    },
    emits: ['update:adapterId', 'update:weight'],
    template: `
        <div class="adapter-picker">
            <el-form-item :label="$t('studio.loraLabel')">
                <el-select
                    :model-value="adapterId"
                    style="width: 100%;"
                    :placeholder="$t('studio.noLora')"
                    @update:model-value="$emit('update:adapterId', $event)"
                >
                    <el-option :label="$t('studio.noLora')" value="" />
                    <el-option
                        v-for="it in items"
                        :key="it.id"
                        :label="it.name || it.id"
                        :value="it.id"
                    />
                </el-select>
                <el-slider
                    v-if="adapterId"
                    :model-value="weight"
                    :min="weightSpec.min"
                    :max="weightSpec.max"
                    :step="weightSpec.step"
                    style="margin-top: 8px;"
                    @update:model-value="$emit('update:weight', $event)"
                />
            </el-form-item>
        </div>
    `,
};

<script setup lang="ts">
type ModelVersion = {
  name?: string;
  size?: string;
  source_type?: string;
  from_version?: string;
};

defineProps<{
  currentModelConfig: { versions?: Record<string, ModelVersion> };
  selectedModel: string;
  selectedDefaultVersion: string;
  capabilityList: { key: string; label: string; value: boolean }[];
  systemInfo: { memory_gb?: number };
  minVersionSizeGB: number;
  memoryProgressColor: string;
  recommendedVersion: { name?: string; size?: string } | null;
  hardwareAdvice: { message?: string } | null;
  hasParamNotes: boolean;
  paramNotesList: { key: string; label: string; note: string }[];
  versionStatusType: (modelId: string, verKey: string) => string;
  versionStatusLabel: (modelId: string, verKey: string) => string;
  isRecommendedVersion: (verKey: string) => boolean;
}>();
</script>

<template>
  <DqInspectorStack aria-label="model-inspector">
    <DqInspectorSection :title="$t('settings.versionStatus')">
      <DqInspectorEmpty v-if="!currentModelConfig.versions">
        {{ $t('settings.noVersions') }}
      </DqInspectorEmpty>
      <DqInspectorList v-else>
        <DqInspectorListItem
          v-for="(ver, verKey) in currentModelConfig.versions"
          :key="verKey"
        >
          <div class="dq-inspector-version__head">
            <span class="dq-inspector-version__name">{{ ver.name || verKey }}</span>
            <span v-if="ver.size" class="dq-inspector-version__size">{{ ver.size }}</span>
          </div>
          <div class="dq-inspector-version__tags">
            <DqTag
              :type="versionStatusType(selectedModel, verKey)"
              size="small"
              effect="plain"
            >
              {{ versionStatusLabel(selectedModel, verKey) }}
            </DqTag>
            <DqTag
              v-if="selectedDefaultVersion === verKey"
              size="small"
              type="info"
              effect="plain"
            >{{ $t('settings.default') }}</DqTag>
            <DqTag
              v-if="isRecommendedVersion(verKey)"
              size="small"
              type="warning"
              effect="plain"
            >{{ $t('settings.recommendedShort') }}</DqTag>
          </div>
          <p
            v-if="ver.source_type === 'derived'"
            class="dq-inspector-footnote"
          >
            {{ $t('settings.from') }}
            {{
              (ver.from_version && currentModelConfig.versions?.[ver.from_version]?.name)
                || ver.from_version
            }}
          </p>
          <p
            v-if="isRecommendedVersion(verKey)"
            class="dq-inspector-highlight"
          >
            <DqIcon><star-filled /></DqIcon>
            {{ $t('settings.recommendedForYourHardware') }}
          </p>
        </DqInspectorListItem>
      </DqInspectorList>
    </DqInspectorSection>

    <DqInspectorSection :title="$t('settings.capabilities')">
      <DqInspectorEmpty v-if="capabilityList.length === 0">
        {{ $t('settings.noCapabilities') }}
      </DqInspectorEmpty>
      <DqInspectorList v-else>
        <DqInspectorListItem
          v-for="cap in capabilityList"
          :key="cap.key"
          class="dq-inspector-cap-row"
          :class="cap.value ? 'is-on' : 'is-off'"
        >
          <DqIcon :size="15" class="dq-inspector-cap-row__icon">
            <component :is="cap.value ? 'check' : 'close'" />
          </DqIcon>
          <span class="dq-inspector-cap-row__label">{{ cap.label }}</span>
          <span class="dq-inspector-cap-row__state">
            {{ cap.value ? $t('settings.featureOn') : $t('settings.featureOff') }}
          </span>
        </DqInspectorListItem>
      </DqInspectorList>
    </DqInspectorSection>

    <DqInspectorSection :title="$t('settings.hardwareCompatibility')">
      <DqInspectorSectionBody>
        <DqInspectorKv
          v-if="systemInfo.memory_gb"
          :label="$t('settings.systemMemory')"
          :value="`${systemInfo.memory_gb.toFixed(1)} GB`"
        />
        <DqProgress
          v-if="systemInfo.memory_gb && minVersionSizeGB > 0"
          class="dq-inspector-progress"
          :percentage="Math.min(100, (minVersionSizeGB / systemInfo.memory_gb) * 100)"
          :show-text="false"
          :stroke-width="5"
          :color="memoryProgressColor"
        />
        <DqInspectorCallout
          v-if="recommendedVersion"
          variant="success"
          :title="$t('settings.recommendedVersion')"
        >
          {{ recommendedVersion.name }}<span v-if="recommendedVersion.size"> · {{ recommendedVersion.size }}</span>
        </DqInspectorCallout>
        <DqInspectorCallout
          v-else-if="currentModelConfig.versions"
          variant="warn"
          :title="$t('settings.memoryInsufficient')"
        />
        <p
          v-if="hardwareAdvice"
          class="dq-inspector-footnote"
        >
          {{ hardwareAdvice.message }}
        </p>
      </DqInspectorSectionBody>
    </DqInspectorSection>

    <DqInspectorSection
      :title="$t('settings.paramNotes')"
      last
    >
      <DqInspectorEmpty v-if="!hasParamNotes">
        {{ $t('settings.noParamNotes') }}
      </DqInspectorEmpty>
      <DqInspectorList v-else>
        <DqInspectorListItem
          v-for="note in paramNotesList"
          :key="note.key"
        >
          <div class="dq-inspector-note__label">{{ note.label }}</div>
          <div class="dq-inspector-note__body">{{ note.note }}</div>
        </DqInspectorListItem>
      </DqInspectorList>
    </DqInspectorSection>
  </DqInspectorStack>
</template>

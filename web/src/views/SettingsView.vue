<script setup lang="ts">
// 설정: config.yaml 의 편집 가능한 값 저장 (다음 작업부터 적용), 계정 변경
import { computed, onMounted, ref } from 'vue'

import { api, ApiError } from '../api/client'
import type { FieldError, ServerStatus, SettingsState, SettingsValues } from '../api/types'
import { errorText, fieldErrorText, numberOrText } from '../format'
import { t, type MessageKey } from '../i18n'
import { showError, showSuccess } from '../notify'
import AccountPanel from '../settings/AccountPanel.vue'
import SettingField from '../settings/SettingField.vue'
import {
  allFields,
  FIELD_TABS,
  getPath,
  isFieldVisible,
  resolveOptions,
  setPath,
  TAB_FIELDS,
  TABS,
  type SettingField as Field,
  type SettingsTab,
} from '../settings/fields'

defineProps<{ status: ServerStatus | null }>()
const emit = defineEmits<{ 'status-changed': []; relogin: [] }>()

const state = ref<SettingsState | null>(null)
const form = ref<SettingsValues | null>(null)
const errors = ref<FieldError[]>([])
const tab = ref<SettingsTab>('stt')
const loading = ref(false)
const saving = ref(false)

const writable = computed(() => state.value?.writable === true)
const tabs = computed(() =>
  TABS.map((value) => ({
    value,
    label: t(`settings.tab.${value}` as MessageKey),
    // 오류가 있는 탭을 아이콘으로 알린다
    icon: tabHasError(value) ? 'mdi-alert-circle-outline' : undefined,
  })),
)
const tabFields = computed(
  () =>
    Object.fromEntries(
      Object.entries(TAB_FIELDS).map(([name, fields]) => [name, fields.map((f) => resolveOptions(f, state.value?.choices))]),
    ) as typeof TAB_FIELDS,
)
// 칸에 붙지 않는 오류(교차 검증: segmentation, audio.vad 등)의 섹션 → 탭 (배너에 탭 이름을 붙인다)
const SECTION_TABS: Record<string, SettingsTab> = {
  stt: 'stt',
  audio: 'audio',
  segmentation: 'segmentation',
  diarization: 'speaker',
  noise_filter: 'speaker',
  postprocess: 'speaker',
  scene: 'quality',
  quality: 'quality',
  output: 'quality',
  logging: 'quality',
  llm: 'llm',
  web: 'web',
}
const FIELD_PATHS = new Set(allFields().map((field) => field.path))

function belongsToField(loc: string): boolean {
  return [...FIELD_PATHS].some((path) => loc === path || loc.startsWith(`${path}.`))
}

const sectionErrors = computed(() =>
  errors.value
    .filter((error) => !belongsToField(error.loc))
    .map((error) => {
      const tabName = SECTION_TABS[error.loc.split('.')[0]]
      const prefix = tabName ? t(`settings.tab.${tabName}` as MessageKey) : error.loc
      return `${prefix}: ${fieldErrorText(error)}`
    }),
)

function fieldsOf(value: SettingsTab): Field[] {
  return value === 'account' ? [] : TAB_FIELDS[value]
}

function errorFor(path: string): string | undefined {
  const found = errors.value.find((error) => error.loc === path || error.loc.startsWith(`${path}.`))
  return found ? fieldErrorText(found) : undefined
}

function tabHasError(value: SettingsTab): boolean {
  if (fieldsOf(value).some((field) => errorFor(field.path) !== undefined)) return true
  return errors.value.some((error) => !belongsToField(error.loc) && SECTION_TABS[error.loc.split('.')[0]] === value)
}

// LLM 탭 배너: 지금 고른 프로바이더의 준비 상태 (구독 CLI 설치, .env 키 설정)
const llmBanner = computed(() => {
  const provider = form.value?.llm.provider
  const readiness = provider ? state.value?.llm[provider] : undefined
  if (!readiness) return null
  const name = readiness.requirement
  if (readiness.kind === 'local') return { type: 'info', text: t('settings.llm.localReady') }
  if (readiness.kind === 'subscription') {
    return readiness.ready
      ? { type: 'success', text: t('settings.llm.cliFound', { name }) }
      : { type: 'warning', text: t('settings.llm.cliMissing', { name }) }
  }
  return readiness.ready
    ? { type: 'success', text: t('settings.llm.keySet', { name }) }
    : { type: 'warning', text: t('settings.llm.keyMissing', { name }) }
})

function valueOf(path: string): unknown {
  return getPath(form.value, path)
}

function update(path: string, value: unknown): void {
  if (form.value) setPath(form.value, path, value)
}

/** 설정 값 복사본. ref 안의 값은 반응형 프록시라 structuredClone 할 수 없어 JSON 으로 복사한다. */
function copyValues(values: SettingsValues): SettingsValues {
  return JSON.parse(JSON.stringify(values)) as SettingsValues
}

async function load(): Promise<void> {
  loading.value = true
  try {
    const loaded = await api.settings()
    state.value = loaded
    form.value = copyValues(loaded.values)
    errors.value = []
  } catch (error) {
    showError(errorText(error))
  } finally {
    loading.value = false
  }
}

/** 숫자 칸은 입력하는 동안 글자로 두었다가 보낼 때 숫자로 바꾼다. 비울 수 있는 칸은 비우면 null. */
function payload(): SettingsValues {
  const values = copyValues(form.value as SettingsValues)
  for (const field of allFields()) {
    if (field.kind !== 'number') continue
    const raw = getPath(values, field.path)
    if (field.nullable && (raw === null || (typeof raw === 'string' && raw.trim() === ''))) {
      setPath(values, field.path, null)
    } else if (typeof raw === 'string' || typeof raw === 'number') {
      setPath(values, field.path, numberOrText(raw))
    }
  }
  return values
}

async function save(): Promise<void> {
  if (!form.value) return
  saving.value = true
  try {
    const saved = await api.saveSettings(payload())
    state.value = saved
    form.value = copyValues(saved.values)
    errors.value = []
    showSuccess(t('settings.saved'))
    emit('status-changed')
  } catch (error) {
    if (error instanceof ApiError && error.code === 'invalid_settings') {
      errors.value = error.errors
      const first = TABS.find(tabHasError)
      if (first) tab.value = first
    }
    showError(errorText(error))
  } finally {
    saving.value = false
  }
}

onMounted(load)
</script>

<template>
  <smartview-page-header :heading="t('nav.settings')" :subtitle="state ? t('settings.subtitle', { path: state.path }) : ''">
    <smartview-button
      slot="actions" data-testid="settings-reload" variant="neutral" icon-name="mdi-refresh" :label="t('common.refresh')"
      :loading="loading" @click="load"
    />
    <smartview-button
      slot="actions" data-testid="settings-save" variant="brand" icon-name="mdi-content-save-outline" :label="t('common.save')"
      :disabled="!writable || !form" :loading="saving" @click="save"
    />
  </smartview-page-header>

  <div v-if="state && form" class="body">
    <smartview-state-notice v-if="!writable" type="warning" :heading="t('settings.readOnly')" :text="t('settings.readOnlyText')" />
    <smartview-hint-banner type="info" :text="t('settings.notice')" />
    <smartview-hint-banner v-for="message in sectionErrors" :key="message" type="error" :text="message" />

    <fieldset class="fields" :disabled="!writable">
      <smartview-tabs data-testid="settings-tabs" :tabs="tabs" :tab="tab" @tab-change="tab = ($event as CustomEvent<[SettingsTab]>).detail[0]">
        <div v-for="name in FIELD_TABS" :key="name" v-bind="{ slot: `tab-${name}` }" class="tab">
          <smartview-hint-banner
            v-if="name === 'speaker'" class="tab-banner" :type="state.hf_token ? 'success' : 'warning'"
            :text="state.hf_token ? t('settings.hfSet') : t('settings.hfMissing')"
          />
          <smartview-hint-banner
            v-if="name === 'llm' && llmBanner" data-testid="llm-readiness" class="tab-banner" :type="llmBanner.type"
            :text="llmBanner.text"
          />
          <div class="grid">
            <SettingField
              v-for="field in tabFields[name].filter((item) => isFieldVisible(item, form))" :key="field.path" :data-testid="`field-${field.path}`" :class="{ wide: field.kind === 'tags' }"
              :field="field" :value="valueOf(field.path)" :error="errorFor(field.path)" @update="update(field.path, $event)"
            />
          </div>
        </div>
        <div slot="tab-account" class="tab">
          <AccountPanel @changed="emit('relogin')" />
        </div>
      </smartview-tabs>
    </fieldset>
  </div>
  <smartview-spinner v-else-if="loading" :alternative-text="t('common.loading')" />
</template>

<style scoped>
.body {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 16px;
}
.fields {
  border: 0;
  margin: 0;
  padding: 0;
  min-width: 0;
}
.tab {
  padding: 16px 0;
}
.tab-banner {
  display: block;
  margin-bottom: 16px;
}
.grid {
  display: grid;
  /* 토글 라벨·스위치·상태 글자가 한 줄에 들어가는 폭 (좁으면 라벨이 위로 올라가 줄이 어긋난다) */
  grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 12px 24px;
  align-items: start;
}
.wide {
  grid-column: 1 / -1;
}
</style>

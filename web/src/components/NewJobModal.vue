<script setup lang="ts">
// 새 작업: 파일을 고르고 처리 옵션을 정한 뒤 올려서 작업으로 등록한다 (업로드는 진행률을 보인다)
import { computed, ref, watch } from 'vue'

import { api, uploadFile } from '../api/client'
import type { JobFormOptions, JobOptions } from '../api/types'
import { errorText, fileSize } from '../format'
import { locale, t, type MessageKey } from '../i18n'
import { showError, showSuccess } from '../notify'

const props = defineProps<{ open: boolean }>()
const emit = defineEmits<{ close: []; created: [] }>()

const MB = 1024 * 1024

const form = ref<JobFormOptions | null>(null)
const options = ref<JobOptions | null>(null)
const file = ref<File | null>(null)
const fileError = ref('')
const maxCpsText = ref('')
const audioTrackText = ref('')
const uploading = ref(false)
const uploaded = ref(0)
let abortUpload: (() => void) | null = null

const percent = computed(() => (file.value && file.value.size > 0 ? Math.round((uploaded.value / file.value.size) * 100) : 0))
const accept = computed(() => form.value?.extensions.join(',') ?? '')
const fileHint = computed(() =>
  form.value
    ? t('newJob.fileHint', { mb: form.value.max_upload_mb, extensions: form.value.extensions.join(' ') })
    : '',
)
const selectedText = computed(() => (file.value ? t('newJob.selected', { name: file.value.name, size: fileSize(file.value.size) }) : ''))

function literalOptions(values: string[]): { value: string; label: string }[] {
  return values.map((value) => ({ value, label: value }))
}
const engineOptions = computed(() => literalOptions(form.value?.choices.engines ?? []))
const sizeOptions = computed(() => literalOptions(form.value?.choices.model_sizes ?? []))
const computeOptions = computed(() => literalOptions(form.value?.choices.compute_types ?? []))
const languageOptions = computed(() =>
  (form.value?.choices.languages ?? []).map((value) => ({
    value,
    label: value === 'auto' ? t('newJob.languageAuto') : value,
  })),
)
const formatOptions = computed(() =>
  (form.value?.choices.formats ?? []).map((value) => ({ value, label: value.toUpperCase() })),
)
const modeOptions = computed(() =>
  (form.value?.choices.modes ?? []).map((value) => ({ value, label: t(`mode.${value}` as MessageKey) })),
)

const cpsError = computed(() => {
  const text = maxCpsText.value.trim()
  if (!text) return ''
  const value = Number(text)
  return Number.isFinite(value) && value > 0 ? '' : t('newJob.invalidCps')
})
const audioTrackError = computed(() => {
  const text = audioTrackText.value.trim()
  return !text || /^\d+$/.test(text) ? '' : t('newJob.invalidAudioTrack')
})
const speakersError = computed(() => {
  const text = options.value?.num_speakers.trim().toLowerCase() ?? 'auto'
  return text === 'auto' || /^[1-9]\d*$/.test(text) ? '' : t('newJob.invalidSpeakers')
})
const formatsError = computed(() => (options.value && options.value.formats.length === 0 ? t('newJob.formatsRequired') : ''))
const ready = computed(
  () =>
    !!file.value && !fileError.value && !cpsError.value && !audioTrackError.value && !speakersError.value && !formatsError.value,
)

async function load(): Promise<void> {
  try {
    const loaded = await api.options()
    form.value = loaded
    options.value = { ...loaded.defaults, formats: [...loaded.defaults.formats] }
    maxCpsText.value = loaded.defaults.max_cps === null ? '' : String(loaded.defaults.max_cps)
    audioTrackText.value = loaded.defaults.audio_track === null ? '' : String(loaded.defaults.audio_track)
  } catch (error) {
    showError(errorText(error))
  }
}

// 열 때마다 서버의 지금 기본값(설정 파일)으로 새로 채운다
watch(
  () => props.open,
  (open) => {
    if (!open) return
    file.value = null
    fileError.value = ''
    uploaded.value = 0
    void load()
  },
  { immediate: true },
)

function onSelect(event: Event): void {
  const [files] = (event as CustomEvent<[File[]]>).detail
  const chosen = files[0] ?? null
  file.value = chosen
  fileError.value = ''
  if (chosen && form.value && chosen.size > form.value.max_upload_mb * MB) {
    fileError.value = t('newJob.fileTooLarge', { mb: form.value.max_upload_mb })
  }
}

function set<K extends keyof JobOptions>(key: K, value: JobOptions[K]): void {
  if (options.value) options.value[key] = value
}

function detail<T>(event: Event): T {
  return (event as CustomEvent<[T]>).detail[0]
}

async function submit(): Promise<void> {
  if (!file.value) {
    fileError.value = t('newJob.fileRequired')
    return
  }
  if (!ready.value || !options.value) return
  const chosen = file.value
  const jobOptions: JobOptions = {
    ...options.value,
    max_cps: maxCpsText.value.trim() ? Number(maxCpsText.value) : null,
    audio_track: audioTrackText.value.trim() ? Number(audioTrackText.value) : null,
    num_speakers: options.value.num_speakers.trim().toLowerCase() || 'auto',
    // 검수 리포트의 메모는 자막 언어가 아니라 지금 화면 언어로 받는다
    review_language: locale.value,
  }
  uploading.value = true
  uploaded.value = 0
  try {
    const upload = uploadFile(chosen, (loaded) => {
      uploaded.value = loaded
    })
    abortUpload = upload.abort
    const result = await upload.promise
    await api.createJob(result.id, jobOptions)
    showSuccess(t('newJob.registered', { name: result.name }))
    emit('created')
    emit('close')
  } catch (error) {
    showError(errorText(error))
  } finally {
    abortUpload = null
    uploading.value = false
  }
}

function cancel(): void {
  // 올리는 중이면 업로드를 멈추고 닫는다 (서버의 반쪽 파일은 서버가 지운다)
  abortUpload?.()
  emit('close')
}
</script>

<template>
  <smartview-modal
    data-testid="new-job-modal" :open="open" :heading="t('newJob.heading')" icon="mdi-file-upload-outline" max-width="880"
    :confirm-text="t('newJob.confirm')" :confirm-disabled="!ready" :loading="uploading" @confirm="submit" @cancel="cancel"
  >
    <!-- 모달은 열릴 때 내용 요소를 오버레이로 옮긴다. 열린 뒤 새로 만든 요소는 옮겨지지 않으므로
         v-if 대신 v-show 로 같은 요소를 보였다 숨긴다 -->
    <div class="form" data-testid="new-job-form">
      <smartview-file-selector
        data-testid="new-job-file" :label="t('newJob.file')" :accept="accept" :hint="selectedText || fileHint"
        :button-label="t('newJob.fileButton')" :drop-text="t('newJob.fileDrop')" :error-message="fileError"
        :disabled="uploading" @select="onSelect"
      />
      <smartview-progress-bar
        v-show="uploading" data-testid="new-job-progress" :value="percent" color="primary" :height="8" text-position="top"
        :text="t('newJob.uploading', { percent })"
      />
      <fieldset v-if="options" class="fields" :disabled="uploading">
        <h3 class="section">{{ t('newJob.sectionStt') }}</h3>
        <div class="grid">
          <smartview-combobox
            data-testid="new-job-engine" :label="t('newJob.engine')" :options="engineOptions" :value="options.engine"
            density="compact" @change="set('engine', detail<string>($event))"
          />
          <smartview-combobox
            data-testid="new-job-size" :label="t('newJob.modelSize')" :options="sizeOptions" :value="options.model_size"
            density="compact" @change="set('model_size', detail<string>($event))"
          />
          <smartview-combobox
            data-testid="new-job-language" :label="t('newJob.language')" :options="languageOptions" :value="options.source_lang"
            searchable density="compact" @change="set('source_lang', detail<string>($event))"
          />
          <smartview-combobox
            data-testid="new-job-compute" :label="t('newJob.computeType')" :hint="t('newJob.computeHint')" :options="computeOptions"
            :value="options.compute_type" density="compact" @change="set('compute_type', detail<string>($event))"
          />
          <smartview-input
            data-testid="new-job-audio-track" :label="t('newJob.audioTrack')" :hint="t('newJob.audioTrackHint')"
            :value="audioTrackText" :error-message="audioTrackError" type="number" density="compact"
            @input="audioTrackText = detail<string>($event)"
          />
        </div>
        <div class="toggles">
          <smartview-checkbox-toggle
            data-testid="new-job-prevad" :label="t('newJob.preVad')" :checked="options.pre_vad"
            @change="set('pre_vad', detail<boolean>($event))"
          />
        </div>
        <h3 class="section">{{ t('newJob.sectionOutput') }}</h3>
        <div class="grid">
          <smartview-checkbox-button-group
            data-testid="new-job-formats" class="wide" :label="t('newJob.formats')" :options="formatOptions" :value="options.formats"
            :error-message="formatsError" @change="set('formats', detail<string[]>($event))"
          />
          <smartview-combobox
            data-testid="new-job-mode" :label="t('newJob.mode')" :options="modeOptions" :value="options.segmentation_mode"
            density="compact" @change="set('segmentation_mode', detail<string>($event))"
          />
          <smartview-input
            data-testid="new-job-cps" :label="t('newJob.maxCps')" :hint="t('newJob.maxCpsHint')" :value="maxCpsText"
            :error-message="cpsError" type="number" density="compact" @input="maxCpsText = detail<string>($event)"
          />
        </div>
        <div class="toggles">
          <smartview-checkbox-toggle
            data-testid="new-job-punctuation" :label="t('newJob.punctuation')" :checked="options.restore_punctuation"
            @change="set('restore_punctuation', detail<boolean>($event))"
          />
          <smartview-checkbox-toggle
            data-testid="new-job-llm-correction" :label="t('newJob.llmCorrection')" :checked="options.llm_correction"
            @change="set('llm_correction', detail<boolean>($event))"
          />
          <smartview-checkbox-toggle
            data-testid="new-job-llm-punctuation" :label="t('newJob.llmPunctuation')" :checked="options.llm_punctuation"
            @change="set('llm_punctuation', detail<boolean>($event))"
          />
          <smartview-checkbox-toggle
            data-testid="new-job-llm-review" :label="t('newJob.llmReview')" :checked="options.llm_review"
            @change="set('llm_review', detail<boolean>($event))"
          />
          <smartview-checkbox-toggle
            data-testid="new-job-report" :label="t('newJob.qualityReport')" :checked="options.quality_report"
            @change="set('quality_report', detail<boolean>($event))"
          />
          <smartview-checkbox-toggle
            data-testid="new-job-words" :label="t('newJob.exportWords')" :checked="options.export_words"
            @change="set('export_words', detail<boolean>($event))"
          />
        </div>
        <h3 class="section">{{ t('newJob.sectionAudio') }}</h3>
        <div class="toggles">
          <smartview-checkbox-toggle
            data-testid="new-job-normalize" :label="t('newJob.normalize')" :checked="options.normalize"
            @change="set('normalize', detail<boolean>($event))"
          />
          <smartview-checkbox-toggle
            data-testid="new-job-denoise" :label="t('newJob.denoise')" :checked="options.denoise"
            @change="set('denoise', detail<boolean>($event))"
          />
          <smartview-checkbox-toggle
            data-testid="new-job-vocals" :label="t('newJob.separateVocals')" :checked="options.separate_vocals"
            @change="set('separate_vocals', detail<boolean>($event))"
          />
          <smartview-checkbox-toggle
            data-testid="new-job-diarize" :label="t('newJob.diarize')" :hint="t('newJob.diarizeHint')" :checked="options.diarize"
            @change="set('diarize', detail<boolean>($event))"
          />
        </div>
        <div v-show="options.diarize" class="grid speakers">
          <smartview-input
            data-testid="new-job-speakers" :label="t('newJob.numSpeakers')"
            :hint="t('newJob.numSpeakersHint')" :value="options.num_speakers" :error-message="speakersError" density="compact"
            @input="set('num_speakers', detail<string>($event))"
          />
        </div>
      </fieldset>
    </div>
  </smartview-modal>
</template>

<style scoped>
.form {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.fields {
  border: 0;
  margin: 0;
  padding: 0;
  min-width: 0;
}
.section {
  margin: 12px 0 8px;
  font-size: 0.95rem;
  font-weight: 600;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 12px 20px;
  align-items: start;
}
.wide {
  grid-column: 1 / -1;
}
/* 토글은 라벨·스위치·상태 글자가 한 줄에 들어갈 만큼 넓은 칸에 둔다 (좁으면 라벨이 위로 올라가 줄이 어긋난다) */
.toggles {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 4px 20px;
  margin-top: 8px;
}
.speakers {
  margin-top: 8px;
}
</style>

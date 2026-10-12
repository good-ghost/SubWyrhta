<script setup lang="ts">
// 작업 상세: 상태·결과 요약·처리 옵션, 출력 파일 내려받기·미리 보기
import { computed, ref, watch } from 'vue'

import { api, downloadHref } from '../api/client'
import type { Job } from '../api/types'
import { durationText, errorText, fileSize, jobErrorText, localTime } from '../format'
import { t } from '../i18n'
import { progressText, STATUS_LOOKS } from '../jobs'
import { showError } from '../notify'

const props = defineProps<{ job: Job | null }>()
const emit = defineEmits<{ close: []; refresh: [] }>()

interface OutputRow {
  id: string
  name: string
}

const previewText = ref('')
const previewName = ref('')
const previewTruncated = ref(false)
const previewLoading = ref(false)

watch(
  () => props.job?.id,
  () => {
    previewText.value = ''
    previewName.value = ''
    previewTruncated.value = false
  },
)

function onOff(value: boolean): string {
  return value ? t('detail.optionOn') : t('detail.optionOff')
}

const statusRows = computed(() => {
  const job = props.job
  if (!job) return []
  const elapsed = job.started_at && job.finished_at ? job.finished_at - job.started_at : null
  return [
    {
      icon: STATUS_LOOKS[job.status].icon.replace(' mdi-spin', ''),
      label: t('detail.status'),
      value: progressText(job),
      chip: STATUS_LOOKS[job.status].color,
    },
    { icon: 'mdi-file-outline', label: t('detail.size'), value: fileSize(job.size) },
    { icon: 'mdi-calendar-plus', label: t('detail.created'), value: localTime(job.created_at) },
    { icon: 'mdi-play-circle-outline', label: t('detail.started'), value: localTime(job.started_at) },
    { icon: 'mdi-flag-checkered', label: t('detail.finished'), value: localTime(job.finished_at) },
    { icon: 'mdi-timer-outline', label: t('detail.elapsed'), value: durationText(elapsed) },
    { icon: 'mdi-refresh', label: t('detail.attempts'), value: String(job.attempts) },
    ...(job.input_deleted ? [{ icon: 'mdi-delete-outline', label: t('detail.inputDeleted'), value: t('common.yes') }] : []),
  ]
})

const resultRows = computed(() => {
  const result = props.job?.result
  if (!result) return []
  const rows: { icon: string; label: string; value: string; chip?: string }[] = [
    { icon: 'mdi-translate', label: t('detail.language'), value: result.language || t('common.none') },
    { icon: 'mdi-clock-outline', label: t('detail.duration'), value: durationText(result.duration_s) },
    { icon: 'mdi-subtitles-outline', label: t('detail.segments'), value: String(result.segments) },
    { icon: 'mdi-robot-outline', label: t('detail.model'), value: result.model || t('common.none') },
  ]
  if (result.num_speakers !== null) {
    rows.push({ icon: 'mdi-account-multiple-outline', label: t('detail.speakers'), value: String(result.num_speakers) })
  }
  const quality = result.quality
  if (quality) {
    rows.push(
      { icon: 'mdi-medal-outline', label: t('detail.grade'), value: quality.grade || t('common.none'), chip: gradeColor(quality.grade) },
      { icon: 'mdi-percent-outline', label: t('detail.confidence'), value: `${(quality.avg_confidence * 100).toFixed(1)}%` },
      {
        icon: 'mdi-alert-outline',
        label: t('detail.issues'),
        value: `${quality.errors} / ${quality.warnings} / ${quality.auto_fixed}`,
      },
    )
  }
  return rows
})

const optionRows = computed(() => {
  const options = props.job?.options
  if (!options) return []
  return [
    {
      icon: 'mdi-microphone-outline',
      label: t('newJob.engine'),
      value: `${options.engine} / ${options.model_size} / ${options.compute_type}`,
    },
    { icon: 'mdi-translate', label: t('newJob.language'), value: options.source_lang === 'auto' ? t('newJob.languageAuto') : options.source_lang },
    { icon: 'mdi-file-document-outline', label: t('newJob.formats'), value: options.formats.map((f) => f.toUpperCase()).join(', ') },
    // 모드 설명까지 넣으면 값 칸에서 잘린다: 모드 이름만
    { icon: 'mdi-content-cut', label: t('newJob.mode'), value: options.segmentation_mode },
    { icon: 'mdi-waveform', label: t('newJob.preVad'), value: onOff(options.pre_vad) },
    // 값 칸이 좁아 오디오 옵션은 항목마다 한 줄 (한 줄로 이으면 잘린다)
    { icon: 'mdi-volume-high', label: t('newJob.normalize'), value: onOff(options.normalize) },
    { icon: 'mdi-waveform', label: t('newJob.denoise'), value: onOff(options.denoise) },
    { icon: 'mdi-account-music-outline', label: t('newJob.separateVocals'), value: onOff(options.separate_vocals) },
    {
      icon: 'mdi-account-voice',
      label: t('newJob.diarize'),
      value: options.diarize ? `${onOff(true)} (${options.num_speakers})` : onOff(false),
    },
  ]
})

const outputs = computed<OutputRow[]>(() => (props.job?.outputs ?? []).map((name) => ({ id: name, name })))
const outputHeaders = computed(() => [
  { title: t('detail.col.file'), key: 'name' },
  {
    title: t('detail.col.actions'), key: 'actions', type: 'actions', width: '110px',
    actions: [
      { name: 'preview', icon: 'mdi-eye-outline', tooltip: t('detail.preview') },
      {
        name: 'download', icon: 'mdi-download', color: 'primary', tooltip: t('detail.download'),
        href: (row: OutputRow) => (props.job ? downloadHref(props.job.id, row.name) : ''),
      },
    ],
  },
])

function gradeColor(grade: string): string {
  if (grade === 'A' || grade === 'B') return 'success'
  if (grade === 'C') return 'warning'
  return grade ? 'error' : 'grey'
}

async function onOutputAction(event: Event): Promise<void> {
  const [{ action, item }] = (event as CustomEvent<[{ action: string; item: OutputRow }]>).detail
  // download 는 링크(href)라 브라우저가 내려받는다 (Content-Disposition: attachment)
  if (action !== 'preview' || !props.job) return
  previewLoading.value = true
  try {
    const preview = await api.preview(props.job.id, item.name)
    previewName.value = preview.name
    previewText.value = preview.text
    previewTruncated.value = preview.truncated
  } catch (error) {
    showError(errorText(error))
  } finally {
    previewLoading.value = false
  }
}
</script>

<template>
  <smartview-modal
    data-testid="job-detail-modal" :open="job !== null" :heading="job ? t('detail.heading', { id: job.id }) + ' · ' + job.name : ''"
    icon="mdi-information-outline" max-width="960" :confirm-text="t('common.refresh')" :cancel-text="t('common.close')" dismissible
    @confirm="emit('refresh')" @cancel="emit('close')"
  >
    <div class="body">
      <smartview-hint-banner v-show="!!job?.error" data-testid="detail-error" type="error" :text="jobErrorText(job?.error ?? null)" />
      <smartview-data-table
        data-testid="detail-outputs" :headers="outputHeaders" :items="outputs" :items-per-page="-1" hide-footer flat
        empty-icon="mdi-file-hidden" :empty-title="t('detail.noOutputs')" @row-action="onOutputAction"
      />
      <smartview-spinner v-show="previewLoading" :alternative-text="t('common.loading')" />
      <div v-show="!!previewName" class="preview" data-testid="detail-preview">
        <div class="preview-head">
          <strong>{{ previewName }}</strong>
          <span v-show="previewTruncated" class="muted">{{ t('detail.previewTruncated') }}</span>
        </div>
        <pre class="preview-text">{{ previewText }}</pre>
      </div>
      <div class="columns">
        <smartview-card :heading="t('detail.status')" icon="mdi-list-status" variant="outlined">
          <smartview-info-list :rows="statusRows" />
        </smartview-card>
        <smartview-card :heading="t('detail.options')" icon="mdi-tune-variant" variant="outlined">
          <smartview-info-list :rows="optionRows" />
        </smartview-card>
      </div>
      <smartview-card v-show="resultRows.length > 0" :heading="t('detail.result')" icon="mdi-chart-box-outline" variant="outlined">
        <smartview-info-list :rows="resultRows" />
      </smartview-card>
    </div>
  </smartview-modal>
</template>

<style scoped>
.body {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.columns {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 12px;
}
.preview-head {
  display: flex;
  gap: 12px;
  align-items: baseline;
  margin-bottom: 6px;
}
.muted {
  opacity: 0.7;
  font-size: 0.85rem;
}
.preview-text {
  margin: 0;
  padding: 12px;
  max-height: 360px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 13px;
  border: 1px solid currentColor;
  border-radius: 4px;
  opacity: 0.9;
}
</style>

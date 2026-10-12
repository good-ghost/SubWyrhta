<script setup lang="ts">
// 작업 대기열: 대기·처리 중·실패·취소 작업, 새 작업 등록, 취소·다시 시도·삭제, 상세
import { computed, ref } from 'vue'

import { api } from '../api/client'
import type { Job, JobStatus, ServerStatus } from '../api/types'
import JobDetailModal from '../components/JobDetailModal.vue'
import NewJobModal from '../components/NewJobModal.vue'
import { usePolling } from '../composables/polling'
import { compactTime, errorText, fileSize, jobErrorText } from '../format'
import { t, type MessageKey } from '../i18n'
import { canCancel, progressPercent, progressText, statusMap } from '../jobs'
import { showError, showSuccess } from '../notify'

defineProps<{ status: ServerStatus | null }>()
const emit = defineEmits<{ 'status-changed': [] }>()

const REFRESH_MS = 3_000
// 표 오류 칸에 보일 최대 글자 수 (넘으면 말줄임)
const ERROR_PREVIEW_CHARS = 80

function shortError(text: string): string {
  const line = text.split('\n')[0]
  return line.length > ERROR_PREVIEW_CHARS ? `${line.slice(0, ERROR_PREVIEW_CHARS)}…` : line
}
type QueueFilter = 'open' | 'queued' | 'running' | 'failed' | 'cancelled' | 'all'
const FILTER_STATUSES: Record<QueueFilter, JobStatus[] | undefined> = {
  open: ['queued', 'running', 'failed', 'cancelled'],
  queued: ['queued'],
  running: ['running'],
  failed: ['failed'],
  cancelled: ['cancelled'],
  all: undefined,
}

const filter = ref<QueueFilter>('open')
const search = ref('')
const jobs = ref<Job[]>([])
const busy = ref<{ id: number; target: string }[]>([])

const { loading, refreshedAt, refresh } = usePolling(async () => {
  try {
    jobs.value = await api.jobs(FILTER_STATUSES[filter.value])
    syncDetail()
  } catch (error) {
    showError(errorText(error))
  }
}, REFRESH_MS)

// 템플릿에 객체 리터럴로 쓰면 다시 그릴 때마다(자동 갱신) 새 값이 요소에 들어가 사용자가 바꾼 값을 되돌린다
const filterValues = computed(() => ({ status: filter.value }))
const failedCount = computed(() => jobs.value.filter((job) => job.status === 'failed' && job.retryable).length)
const filters = computed(() => [
  {
    key: 'status',
    label: t('queue.filter'),
    items: (Object.keys(FILTER_STATUSES) as QueueFilter[]).map((value) => ({
      title: t(`queue.filter.${value}` as MessageKey),
      value,
    })),
  },
])
const headers = computed(() => [
  {
    title: t('queue.col.name'), key: 'name', type: 'name', maxWidth: '320px',
    subKey: 'sizeText', value: (job: Job) => job.name,
  },
  { title: t('queue.col.status'), key: 'status', type: 'status', width: '130px', statusMap: statusMap() },
  {
    title: t('queue.col.progress'), key: 'progress', type: 'score', width: '280px',
    value: (job: Job) => progressPercent(job),
    scoreText: (job: Job) => (job.status === 'running' ? progressText(job) : ''),
    scoreColor: 'primary',
  },
  { title: t('queue.col.attempts'), key: 'attempts', type: 'number', width: '96px' },
  { title: t('queue.col.created'), key: 'created_at', width: '120px', value: (job: Job) => compactTime(job.created_at) },
  // 표에는 앞부분만 (전체 문구는 상세 창)
  { title: t('queue.col.error'), key: 'error', maxWidth: '260px', value: (job: Job) => shortError(jobErrorText(job.error)) },
  {
    title: t('queue.col.actions'), key: 'actions', type: 'actions', width: '150px',
    actions: [
      { name: 'detail', icon: 'mdi-information-outline', tooltip: t('queue.detail') },
      { name: 'cancel', icon: 'mdi-stop-circle-outline', color: 'warning', tooltip: t('queue.cancel'), visible: canCancel },
      { name: 'retry', icon: 'mdi-refresh', color: 'primary', tooltip: t('queue.retry'), visible: (job: Job) => job.retryable },
      {
        name: 'delete', icon: 'mdi-delete-outline', color: 'error', tooltip: t('queue.delete'),
        visible: (job: Job) => job.status !== 'running',
      },
    ],
  },
])
const rows = computed(() => jobs.value.map((job) => ({ ...job, sizeText: fileSize(job.size) })))

async function onFilterChange(event: Event): Promise<void> {
  const [{ value }] = (event as CustomEvent<[{ key: string; value: string }]>).detail
  filter.value = value as QueueFilter
  await refresh()
}

// 확인 창 (실행 중 취소·삭제)
const confirmJob = ref<Job | null>(null)
const confirmKind = ref<'cancel' | 'delete' | null>(null)
const confirming = ref(false)
const promptOpen = computed(() => confirmJob.value !== null && confirmKind.value !== null)
const promptHeading = computed(() => (confirmKind.value === 'cancel' ? t('queue.cancelHeading') : t('queue.deleteHeading')))
const promptMessage = computed(() => {
  const name = confirmJob.value?.name ?? ''
  return confirmKind.value === 'cancel' ? t('queue.cancelMessage', { name }) : t('queue.deleteMessage', { name })
})
const promptWarning = computed(() => (confirmKind.value === 'cancel' ? t('queue.cancelWarning') : t('queue.deleteWarning')))
const promptConfirm = computed(() => (confirmKind.value === 'cancel' ? t('queue.cancelConfirm') : t('queue.deleteConfirm')))

function ask(kind: 'cancel' | 'delete', job: Job): void {
  confirmKind.value = kind
  confirmJob.value = job
}

function closePrompt(): void {
  confirmJob.value = null
  confirmKind.value = null
}

async function runAction(job: Job, action: string, call: () => Promise<unknown>, message: string): Promise<void> {
  busy.value = [{ id: job.id, target: action }]
  try {
    await call()
    showSuccess(message)
    await refresh()
    emit('status-changed')
  } catch (error) {
    showError(errorText(error))
  } finally {
    busy.value = []
  }
}

async function confirmPrompt(): Promise<void> {
  const job = confirmJob.value
  const kind = confirmKind.value
  if (!job || !kind) return
  confirming.value = true
  try {
    if (kind === 'cancel') await runAction(job, 'cancel', () => api.cancel(job.id), t('queue.cancelled'))
    else await runAction(job, 'delete', () => api.remove(job.id), t('queue.deleted'))
  } finally {
    confirming.value = false
    closePrompt()
  }
}

async function onRowAction(event: Event): Promise<void> {
  const [{ action, item }] = (event as CustomEvent<[{ action: string; item: Job }]>).detail
  const job = jobs.value.find((candidate) => candidate.id === item.id) ?? item
  if (action === 'detail') {
    detail.value = job
  } else if (action === 'cancel') {
    // 대기 중인 작업은 바로 취소한다. 실행 중이면 지금까지의 처리를 멈추므로 확인한다
    if (job.status === 'running') ask('cancel', job)
    else await runAction(job, 'cancel', () => api.cancel(job.id), t('queue.cancelled'))
  } else if (action === 'retry') {
    await runAction(job, 'retry', () => api.retry(job.id), t('queue.retriedOne'))
  } else if (action === 'delete') {
    ask('delete', job)
  }
}

const retrying = ref(false)
async function retryFailed(): Promise<void> {
  retrying.value = true
  try {
    const { count } = await api.retryFailed()
    showSuccess(t('queue.retried', { count }))
    await refresh()
    emit('status-changed')
  } catch (error) {
    showError(errorText(error))
  } finally {
    retrying.value = false
  }
}

// 새 작업·상세 모달
const newJobOpen = ref(false)
const detail = ref<Job | null>(null)

/** 자동 갱신 때 열린 상세의 내용도 맞춘다 */
function syncDetail(): void {
  if (!detail.value) return
  const fresh = jobs.value.find((job) => job.id === detail.value?.id)
  if (fresh) detail.value = fresh
}

async function reloadDetail(): Promise<void> {
  if (!detail.value) return
  try {
    detail.value = await api.job(detail.value.id)
  } catch (error) {
    showError(errorText(error))
  }
}

async function onCreated(): Promise<void> {
  await refresh()
  emit('status-changed')
}

function onRowClick(event: Event): void {
  const [{ item }] = (event as CustomEvent<[{ item: Job }]>).detail
  detail.value = jobs.value.find((job) => job.id === item.id) ?? item
}
</script>

<template>
  <smartview-page-header
    :heading="t('nav.queue')" :subtitle="t('queue.subtitle')" :action-text="t('queue.newJob')" action-icon="mdi-plus"
    @action="newJobOpen = true"
  >
    <smartview-refreshed-at slot="actions" :time="refreshedAt" />
    <smartview-button
      slot="actions" data-testid="retry-failed" variant="neutral" icon-name="mdi-refresh" :label="t('queue.retryFailed')"
      :disabled="failedCount === 0" :loading="retrying" @click="retryFailed"
    />
  </smartview-page-header>

  <smartview-filter-bar
    class="bar" data-testid="queue-filter" :search-label="t('queue.search')" :filters="filters" :values="filterValues"
    :loading="loading" @filter-change="onFilterChange" @search-change="search = ($event as CustomEvent<[string]>).detail[0]"
    @refresh="refresh"
  />
  <smartview-data-table
    data-testid="queue-table" :headers="headers" :items="rows" :search="search" :busy="busy" :loading="loading" row-clickable
    empty-icon="mdi-tray" :empty-title="t('queue.empty')" :empty-hint="t('queue.emptyHint')" @row-action="onRowAction"
    @row-click="onRowClick"
  />

  <NewJobModal :open="newJobOpen" @close="newJobOpen = false" @created="onCreated" />
  <JobDetailModal :job="detail" @close="detail = null" @refresh="reloadDetail" />

  <smartview-prompt
    data-testid="queue-prompt" :open="promptOpen" :heading="promptHeading" :message="promptMessage" :warning="promptWarning"
    :confirm-text="promptConfirm" :loading="confirming" @confirm="confirmPrompt" @cancel="closePrompt"
  />
</template>

<style scoped>
.bar {
  margin: 16px 0;
}
</style>

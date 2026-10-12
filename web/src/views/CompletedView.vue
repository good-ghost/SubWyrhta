<script setup lang="ts">
// 완료 목록: 자막을 만든 작업 — 결과 요약, 출력 파일 미리 보기·내려받기(상세), 자막 검수, 삭제, 끝난 작업 모두 지우기
import { computed, ref } from 'vue'

import { api } from '../api/client'
import type { Job, ServerStatus } from '../api/types'
import JobDetailModal from '../components/JobDetailModal.vue'
import { usePolling } from '../composables/polling'
import { durationText, errorText, isoFromEpoch } from '../format'
import { t } from '../i18n'
import { showError, showSuccess } from '../notify'
import { hasSubtitle } from '../review'
import ReviewView from './ReviewView.vue'

const props = defineProps<{ status: ServerStatus | null }>()
const emit = defineEmits<{ 'status-changed': [] }>()

// 완료 기록은 자주 바뀌지 않는다
const REFRESH_MS = 10_000
const GRADE_COLORS: Record<string, string> = { A: 'success', B: 'success', C: 'warning', D: 'error', F: 'error' }
// 처음 정렬 (상수: 자동 갱신 때 사용자가 고른 정렬을 되돌리지 않게)
const INITIAL_SORT = [{ key: 'finished_at', order: 'desc' }]

const search = ref('')
const jobs = ref<Job[]>([])
const busy = ref<{ id: number; target: string }[]>([])

const { loading, refreshedAt, refresh } = usePolling(async () => {
  try {
    jobs.value = await api.jobs(['done'])
  } catch (error) {
    showError(errorText(error))
  }
}, REFRESH_MS)

// 끝난 작업(완료·실패·취소) 수: 지우기 확인 문구에 쓴다
const finishedCount = computed(() => {
  const queue = props.status?.queue
  return queue ? queue.done + queue.failed + queue.cancelled : jobs.value.length
})

const rows = computed(() =>
  jobs.value.map((job) => ({
    ...job,
    language: job.result?.language ?? '',
    durationText: durationText(job.result?.duration_s),
    segments: job.result?.segments ?? null,
    grade: job.result?.quality?.grade ?? '',
    formats: job.outputs.map((name) => name.split('.').pop()?.toUpperCase() ?? name),
  })),
)
const headers = computed(() => [
  { title: t('completed.col.name'), key: 'name', type: 'name', maxWidth: '320px' },
  { title: t('completed.col.language'), key: 'language', width: '80px' },
  { title: t('completed.col.duration'), key: 'durationText', width: '90px' },
  { title: t('completed.col.segments'), key: 'segments', type: 'number', width: '90px' },
  {
    title: t('completed.col.grade'), key: 'grade', type: 'status', width: '90px',
    statusMap: Object.fromEntries(Object.entries(GRADE_COLORS).map(([grade, color]) => [grade, { color, label: grade }])),
  },
  { title: t('completed.col.outputs'), key: 'formats', type: 'chips', maxWidth: '220px' },
  {
    title: t('completed.col.finished'), key: 'finished_at', type: 'date', width: '160px',
    value: (job: Job) => isoFromEpoch(job.finished_at),
  },
  {
    title: t('completed.col.actions'), key: 'actions', type: 'actions', width: '140px',
    actions: [
      { name: 'detail', icon: 'mdi-information-outline', color: 'primary', tooltip: t('queue.detail') },
      {
        // SRT 출력이 있는 작업만 검수할 수 있다
        name: 'review', icon: 'mdi-file-document-edit-outline', color: 'primary', tooltip: t('completed.review'),
        visible: (job: Job) => hasSubtitle(job.outputs),
      },
      { name: 'delete', icon: 'mdi-delete-outline', color: 'error', tooltip: t('queue.delete') },
    ],
  },
])

const detail = ref<Job | null>(null)
// 검수 중인 작업: 있으면 목록 대신 검수 화면을 보인다
const reviewJob = ref<Job | null>(null)
const deleteJob = ref<Job | null>(null)
const deleting = ref(false)

function onRowAction(event: Event): void {
  const [{ action, item }] = (event as CustomEvent<[{ action: string; item: Job }]>).detail
  const job = jobs.value.find((candidate) => candidate.id === item.id) ?? item
  if (action === 'detail') detail.value = job
  else if (action === 'review') reviewJob.value = job
  else if (action === 'delete') deleteJob.value = job
}

function onRowClick(event: Event): void {
  const [{ item }] = (event as CustomEvent<[{ item: Job }]>).detail
  detail.value = jobs.value.find((job) => job.id === item.id) ?? item
}

async function confirmDelete(): Promise<void> {
  const job = deleteJob.value
  if (!job) return
  deleting.value = true
  busy.value = [{ id: job.id, target: 'delete' }]
  try {
    await api.remove(job.id)
    showSuccess(t('queue.deleted'))
    deleteJob.value = null
    await refresh()
    emit('status-changed')
  } catch (error) {
    showError(errorText(error))
  } finally {
    deleting.value = false
    busy.value = []
  }
}

async function closeReview(): Promise<void> {
  reviewJob.value = null
  // 자막을 고치면 출력 목록에 원본(*.orig.srt)이 늘어난다
  await refresh()
}

async function reloadDetail(): Promise<void> {
  if (!detail.value) return
  try {
    detail.value = await api.job(detail.value.id)
  } catch (error) {
    showError(errorText(error))
  }
}

const clearOpen = ref(false)
const clearing = ref(false)
async function clearFinished(): Promise<void> {
  clearing.value = true
  try {
    const { count } = await api.clearFinished()
    showSuccess(t('completed.cleared', { count }))
    clearOpen.value = false
    await refresh()
    emit('status-changed')
  } catch (error) {
    showError(errorText(error))
  } finally {
    clearing.value = false
  }
}
</script>

<template>
  <ReviewView v-if="reviewJob" :job="reviewJob" @back="closeReview" />
  <template v-else>
    <smartview-page-header :heading="t('nav.completed')" :subtitle="t('completed.subtitle')">
      <smartview-refreshed-at slot="actions" :time="refreshedAt" />
      <smartview-button
        slot="actions" data-testid="clear-finished" variant="destructive-text" icon-name="mdi-delete-sweep-outline"
        :label="t('completed.clear')" :disabled="finishedCount === 0" @click="clearOpen = true"
      />
    </smartview-page-header>

    <smartview-filter-bar
      class="bar" data-testid="completed-filter" :search-label="t('completed.search')" :loading="loading"
      @search-change="search = ($event as CustomEvent<[string]>).detail[0]" @refresh="refresh"
    />
    <smartview-data-table
      data-testid="completed-table" :headers="headers" :items="rows" :search="search" :busy="busy" :loading="loading"
      :sort-by="INITIAL_SORT" row-clickable empty-icon="mdi-subtitles-outline" :empty-title="t('completed.empty')"
      @row-action="onRowAction" @row-click="onRowClick"
    />

    <JobDetailModal :job="detail" @close="detail = null" @refresh="reloadDetail" />

    <smartview-prompt
      data-testid="delete-prompt" :open="deleteJob !== null" :heading="t('queue.deleteHeading')"
      :message="t('queue.deleteMessage', { name: deleteJob?.name ?? '' })" :warning="t('queue.deleteWarning')"
      :confirm-text="t('queue.deleteConfirm')" :loading="deleting" @confirm="confirmDelete" @cancel="deleteJob = null"
    />
    <smartview-prompt
      data-testid="clear-prompt" :open="clearOpen" :heading="t('completed.clearHeading')"
      :message="t('completed.clearMessage', { count: finishedCount })" :warning="t('completed.clearWarning')"
      :confirm-text="t('completed.clearConfirm')" :loading="clearing" @confirm="clearFinished" @cancel="clearOpen = false"
    />
  </template>
</template>

<style scoped>
.bar {
  margin: 16px 0;
}
</style>

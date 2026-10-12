<script setup lang="ts">
// 대시보드: 워커·큐 상태, 현재 작업 진행, 시스템(디바이스·패키지·ffmpeg·HF_TOKEN·디스크), 기본 처리 설정
import { computed, ref } from 'vue'

import { api } from '../api/client'
import type { ServerStatus } from '../api/types'
import { usePolling } from '../composables/polling'
import { durationText, errorText, fileSize } from '../format'
import { t, type MessageKey } from '../i18n'
import { progressPercent, progressText } from '../jobs'
import { showError } from '../notify'
import { routeHref } from '../router'

defineProps<{ status: ServerStatus | null }>()
const emit = defineEmits<{ 'status-changed': [] }>()

const REFRESH_MS = 5_000

const current = ref<ServerStatus | null>(null)
const now = ref(Date.now())
const { loading, refreshedAt, refresh } = usePolling(async () => {
  try {
    current.value = await api.status()
    now.value = Date.now()
  } catch (error) {
    showError(errorText(error))
  }
}, REFRESH_MS)

const workerValue = computed(() => {
  const status = current.value
  if (!status) return ''
  if (!status.worker_alive) return t('dashboard.workerStopped')
  return status.current ? t('dashboard.workerBusy') : t('dashboard.workerIdle')
})
const runningJob = computed(() => {
  const job = current.value?.current
  if (!job) return null
  return { status: 'running' as const, stage: job.stage, progress: job.progress, name: job.name, started_at: job.started_at }
})
const currentRows = computed(() => {
  const job = runningJob.value
  if (!job) return []
  return [
    { icon: 'mdi-file-video-outline', label: t('dashboard.currentFile'), value: job.name },
    { icon: 'mdi-progress-clock', label: t('dashboard.currentStage'), value: progressText(job) },
    { icon: 'mdi-timer-outline', label: t('dashboard.elapsed'), value: durationText((now.value / 1000) - job.started_at) },
  ]
})
const systemRows = computed(() => {
  const status = current.value
  if (!status) return []
  const device = status.system.device
  const rows: { icon: string; label: string; value: string; chip?: string }[] = [
    {
      icon: 'mdi-chip',
      label: t('dashboard.devices'),
      value: device?.available_devices.join(', ') ?? t('common.none'),
    },
    { icon: 'mdi-fire', label: t('dashboard.torch'), value: device?.torch_version ?? t('common.missing') },
  ]
  if (device?.cuda_device_name) rows.push({ icon: 'mdi-expansion-card', label: t('dashboard.gpu'), value: device.cuda_device_name })
  if (device?.cuda_vram_gb) rows.push({ icon: 'mdi-memory', label: t('dashboard.vram'), value: `${device.cuda_vram_gb} GB` })
  rows.push(
    {
      icon: 'mdi-filmstrip',
      label: t('dashboard.ffmpeg'),
      value: status.system.ffmpeg ? t('common.installed') : t('common.missing'),
      chip: status.system.ffmpeg ? 'success' : 'error',
    },
    {
      icon: 'mdi-key-outline',
      label: t('dashboard.hfToken'),
      value: status.hf_token ? t('dashboard.hfSet') : t('dashboard.hfMissing'),
      chip: status.hf_token ? 'success' : 'warning',
    },
    // info-list 는 이름 칸을 줄이지 않고 값 칸만 줄인다. 이름에 전체 용량을 붙이면 값이 잘려
    // 사라지므로 이름은 짧게 두고 여유·전체를 행으로 나눈다
    { icon: 'mdi-harddisk', label: t('dashboard.diskFree'), value: fileSize(status.disk_free) },
    { icon: 'mdi-database-outline', label: t('dashboard.diskTotal'), value: fileSize(status.disk_total) },
  )
  return rows
})
const packageRows = computed(() =>
  Object.entries(current.value?.system.packages ?? {}).map(([name, installed]) => ({
    icon: 'mdi-package-variant-closed',
    label: t(`package.${name}` as MessageKey),
    value: installed ? t('common.installed') : t('common.missing'),
    chip: installed ? 'success' : 'grey',
  })),
)
const defaultRows = computed(() => {
  const defaults = current.value?.defaults
  if (!defaults) return []
  return [
    { icon: 'mdi-microphone-outline', label: t('dashboard.engine'), value: defaults.engine },
    { icon: 'mdi-database-outline', label: t('dashboard.modelSize'), value: defaults.model_size },
    { icon: 'mdi-chip', label: t('dashboard.device'), value: defaults.device },
    { icon: 'mdi-calculator-variant-outline', label: t('dashboard.computeType'), value: defaults.compute_type },
    { icon: 'mdi-translate', label: t('dashboard.language'), value: defaults.source_lang },
    { icon: 'mdi-file-document-outline', label: t('dashboard.format'), value: defaults.format.toUpperCase() },
    { icon: 'mdi-content-cut', label: t('dashboard.mode'), value: defaults.mode },
  ]
})

function onCardNavigate(event: Event): void {
  const [{ href }] = (event as CustomEvent<[{ href: string }]>).detail
  event.preventDefault()
  window.location.hash = href
}

async function onRefresh(): Promise<void> {
  await refresh()
  emit('status-changed')
}
</script>

<template>
  <smartview-page-header :heading="t('nav.dashboard')" :subtitle="t('dashboard.subtitle')">
    <smartview-refreshed-at slot="actions" :time="refreshedAt" />
    <smartview-button
      slot="actions" data-testid="refresh" variant="neutral" icon-name="mdi-refresh" :label="t('common.refresh')"
      :loading="loading" @click="onRefresh"
    />
  </smartview-page-header>

  <div v-if="current" class="stats">
    <smartview-stat-card
      data-testid="stat-worker" :label="t('dashboard.worker')" :value="workerValue"
      :caption="runningJob ? runningJob.name : ''" icon="mdi-cog-play-outline"
      :color="current.worker_alive ? 'primary' : 'error'" :accent="current.worker_alive ? 'blue' : 'red'" :index="0"
    />
    <smartview-stat-card
      data-testid="stat-queued" :label="t('dashboard.queued')" :value="current.queue.queued" icon="mdi-tray-full"
      color="info" accent="cyan" :index="1" :href="routeHref('queue')" @navigate="onCardNavigate"
    />
    <smartview-stat-card
      data-testid="stat-done" :label="t('dashboard.done')" :value="current.queue.done" icon="mdi-check-all"
      color="success" accent="green" :index="2" :href="routeHref('completed')" @navigate="onCardNavigate"
    />
    <smartview-stat-card
      data-testid="stat-failed" :label="t('dashboard.failed')" :value="current.queue.failed" icon="mdi-alert-circle-outline"
      color="error" accent="red" :index="3" :href="routeHref('queue')" @navigate="onCardNavigate"
    />
  </div>

  <div v-if="current" class="cards">
    <smartview-card data-testid="current-card" :heading="t('dashboard.currentCard')" icon="mdi-progress-clock">
      <template v-if="runningJob">
        <smartview-progress-bar
          class="bar" :value="progressPercent(runningJob) ?? 0" color="primary" :height="8" text-position="top"
          :text="progressText(runningJob)"
        />
        <smartview-info-list :rows="currentRows" />
      </template>
      <smartview-empty-state v-else icon="mdi-sleep" :heading="t('dashboard.noCurrent')" />
    </smartview-card>
    <smartview-card data-testid="system-card" :heading="t('dashboard.systemCard')" icon="mdi-server-outline">
      <smartview-hint-banner v-if="current.system_state === 'checking'" type="info" :text="t('dashboard.checking')" />
      <smartview-hint-banner v-else-if="current.system_state === 'failed'" type="warning" :text="t('dashboard.checkFailed')" />
      <smartview-info-list :rows="systemRows" />
    </smartview-card>
    <smartview-card data-testid="packages-card" :heading="t('dashboard.packagesCard')" icon="mdi-package-variant">
      <smartview-info-list :rows="packageRows" />
    </smartview-card>
    <smartview-card data-testid="defaults-card" :heading="t('dashboard.defaultsCard')" icon="mdi-tune-variant">
      <smartview-info-list :rows="defaultRows" />
    </smartview-card>
  </div>
  <smartview-spinner v-else-if="loading" :alternative-text="t('common.loading')" />
</template>

<style scoped>
.stats {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 16px;
  margin: 16px 0;
}
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
  gap: 16px;
}
.bar {
  display: block;
  margin-bottom: 12px;
}
</style>

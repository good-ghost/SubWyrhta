<script setup lang="ts">
// 앱 셸: 사이드 메뉴(대시보드·작업 대기열·완료 목록·로그·설정), 상단 바 워커 상태
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

import { api } from '../api/client'
import type { ServerStatus } from '../api/types'
import { t, type MessageKey } from '../i18n'
import { route, routeHref, ROUTES, type RouteName } from '../router'
import CompletedView from '../views/CompletedView.vue'
import DashboardView from '../views/DashboardView.vue'
import LogsView from '../views/LogsView.vue'
import QueueView from '../views/QueueView.vue'
import SettingsView from '../views/SettingsView.vue'

const emit = defineEmits<{ logout: []; relogin: [] }>()

// 상단 바 상태 칩 갱신 간격
const STATUS_INTERVAL_MS = 10_000

const ICONS: Record<RouteName, string> = {
  dashboard: 'mdi-view-dashboard-outline',
  queue: 'mdi-tray-full',
  completed: 'mdi-check-all',
  logs: 'mdi-text-box-search-outline',
  settings: 'mdi-cog-outline',
}
const VIEWS = {
  dashboard: DashboardView,
  queue: QueueView,
  completed: CompletedView,
  logs: LogsView,
  settings: SettingsView,
} as const

const status = ref<ServerStatus | null>(null)
const statusFailed = ref(false)
let timer: number | undefined

const items = computed(() =>
  ROUTES.map((name) => ({ key: name, label: t(`nav.${name}` as MessageKey), icon: ICONS[name], href: routeHref(name) })),
)
const footerItems = computed(() => [{ key: 'logout', label: t('nav.logout'), icon: 'mdi-logout', color: 'error' }])
const breadcrumbs = computed(() => [
  { label: t('app.name'), href: routeHref('dashboard') },
  { label: t(`nav.${route.value}` as MessageKey) },
])
// 칩 색: 워커가 돌면 ok, 서버가 응답하지 않거나 워커가 멈췄으면 error
const shellStatus = computed(() => {
  if (statusFailed.value) return 'error'
  if (!status.value) return 'unknown'
  return status.value.worker_alive ? 'ok' : 'error'
})
const statusLabel = computed(() => {
  if (statusFailed.value || (status.value && !status.value.worker_alive)) return t('shell.status.error')
  if (!status.value) return t('shell.status.unknown')
  return status.value.current ? t('shell.status.busy') : t('shell.status.idle')
})

async function refreshStatus(): Promise<void> {
  try {
    status.value = await api.status()
    statusFailed.value = false
  } catch {
    // 401 은 클라이언트가 로그인 화면으로 돌린다. 나머지는 상태 칩으로만 알린다
    statusFailed.value = true
  }
}

function onNavigate(event: Event): void {
  const [{ href }] = (event as CustomEvent<[{ href?: string }]>).detail
  event.preventDefault()
  if (href) window.location.hash = href
}

function onFooterAction(event: Event): void {
  const [{ key }] = (event as CustomEvent<[{ key: string }]>).detail
  if (key === 'logout') emit('logout')
}

onMounted(() => {
  void refreshStatus()
  timer = window.setInterval(() => void refreshStatus(), STATUS_INTERVAL_MS)
})
onBeforeUnmount(() => window.clearInterval(timer))
</script>

<template>
  <smartview-app-shell
    class="shell"
    :brand-name="t('app.name')"
    brand-icon="mdi-subtitles-outline"
    :version="status?.version ?? ''"
    :current="route"
    :status="shellStatus"
    :status-label="statusLabel"
    :items="items"
    :footer-items="footerItems"
    :breadcrumbs="breadcrumbs"
    @navigate="onNavigate"
    @footer-action="onFooterAction"
  >
    <main class="page">
      <component :is="VIEWS[route]" :status="status" @status-changed="refreshStatus" @relogin="emit('relogin')" />
    </main>
  </smartview-app-shell>
</template>

<style scoped>
.shell {
  height: 100vh;
}
.page {
  padding: 16px 24px 24px;
  box-sizing: border-box;
}
</style>

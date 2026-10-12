<script setup lang="ts">
// 설정 필드 하나를 vue-smartview 입력 요소로 그린다 (종류는 fields.ts)
import { computed } from 'vue'

import { t, type MessageKey } from '../i18n'
import type { FieldOption, SettingField } from './fields'

const props = defineProps<{ field: SettingField; value: unknown; error?: string }>()
const emit = defineEmits<{ update: [value: unknown] }>()

function optionLabel(option: FieldOption): string {
  return option.literal ? option.label : t(option.label as MessageKey)
}

const label = computed(() => t(props.field.label))
const hint = computed(() => (props.field.hint ? t(props.field.hint) : ''))
const suffix = computed(() => (props.field.suffix ? t(props.field.suffix) : ''))
// 글자·숫자 칸의 값 (배열 필드는 이 값을 쓰지 않는다)
const text = computed(() =>
  typeof props.value === 'string' || typeof props.value === 'number' ? String(props.value) : '',
)
const comboOptions = computed(() => (props.field.options ?? []).map((o) => ({ value: o.value, label: optionLabel(o) })))

function detail<T>(event: Event): T {
  return (event as CustomEvent<[T]>).detail[0]
}
</script>

<template>
  <smartview-input
    v-if="field.kind === 'text' || field.kind === 'number'"
    :label="label" :hint="hint" :suffix="suffix" :value="text" :error-message="error ?? ''"
    :type="field.kind === 'number' ? 'number' : 'text'" density="compact" @input="emit('update', detail<string>($event))"
  />
  <smartview-checkbox-toggle
    v-else-if="field.kind === 'toggle'" :label="label" :hint="hint" :checked="value === true"
    :error-message="error ?? ''" @change="emit('update', detail<boolean>($event))"
  />
  <smartview-combobox
    v-else-if="field.kind === 'select'" :label="label" :hint="hint" :options="comboOptions" :value="text"
    :searchable="field.searchable === true"
    :error-message="error ?? ''" density="compact" @change="emit('update', detail<string>($event))"
  />
  <smartview-multi-picklist
    v-else-if="field.kind === 'tags'" :label="label" :hint="hint" :value="value" :error-message="error ?? ''"
    @change="emit('update', detail<string[]>($event))"
  />
</template>

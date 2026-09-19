<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import type { EChartsType } from 'echarts/core'
import type { ChartDefinition } from '../types'

const props = defineProps<{ chart: ChartDefinition }>()
const chartElement = ref<HTMLElement>()
const loadError = ref(false)
let instance: EChartsType | undefined
let resizeObserver: ResizeObserver | undefined

const dashboardItems = computed(() => {
  const value = props.chart.option || {}
  const money = (number: unknown) => Number(number || 0).toLocaleString('zh-CN', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 })
  const percent = (number: unknown) => number == null ? '—' : `${(Number(number) * 100).toFixed(1)}%`
  return [
    { label: '总 GMV', value: money(value.gmv_usd), tone: 'blue' },
    { label: '客单价', value: money(value.avg_order_value_usd), tone: 'cyan' },
    { label: '复购率', value: percent(value.repeat_purchase_rate), tone: 'purple' },
    { label: '流失率', value: percent(value.churn_rate), tone: 'orange' },
  ]
})

async function render() {
  instance?.dispose()
  instance = undefined
  loadError.value = false
  if (props.chart.type === 'dashboard' || !chartElement.value) return
  try {
    const { createChart } = await import('../lib/chartRuntime')
    if (!chartElement.value) return
    instance = createChart(chartElement.value)
    const isPie = props.chart.type === 'pie'
    instance.setOption({
      color: ['#4f6ef7', '#36cfc9', '#9b6df3', '#f6bd16', '#e8684a', '#5ad8a6', '#f759ab'],
      animationDuration: 500,
      tooltip: { trigger: isPie ? 'item' : 'axis', confine: true },
      legend: isPie
        ? { type: 'scroll', orient: 'vertical', left: 4, top: 'middle' }
        : { type: 'scroll', bottom: 0 },
      grid: isPie ? undefined : { left: 16, right: 20, top: 24, bottom: 48, containLabel: true },
      ...props.chart.option,
    }, true)
  } catch (_) {
    loadError.value = true
  }
}

onMounted(async () => {
  await nextTick()
  await render()
  if (chartElement.value) {
    resizeObserver = new ResizeObserver(() => instance?.resize())
    resizeObserver.observe(chartElement.value)
  }
})
watch(() => props.chart, render, { deep: true })
onUnmounted(() => { resizeObserver?.disconnect(); instance?.dispose() })
</script>

<template>
  <article class="chart-card">
    <header><strong>{{ chart.title || '数据图表' }}</strong><span>{{ chart.type === 'dashboard' ? 'KPI' : chart.type.toUpperCase() }}</span></header>
    <div v-if="chart.type==='dashboard'" class="kpi-grid">
      <div v-for="item in dashboardItems" :key="item.label" :class="['kpi',item.tone]"><strong>{{ item.value }}</strong><span>{{ item.label }}</span></div>
    </div>
    <div v-else ref="chartElement" class="chart-canvas" role="img" :aria-label="chart.title || '数据图表'"></div>
    <div v-if="loadError" class="chart-error">图表暂时无法加载，原始数据仍可在下方查看。</div>
  </article>
</template>

<style scoped>
.chart-card{overflow:hidden;border:1px solid #e3e8f1;border-radius:12px;background:#fff}.chart-card header{height:48px;padding:0 15px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #edf0f5}.chart-card header strong{font-size:13px;color:#253047}.chart-card header span{font-size:10px;color:#7e8bea;background:#eef2ff;padding:3px 7px;border-radius:9px}.chart-canvas{height:330px;width:100%}.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;padding:22px 16px}.kpi{padding:16px 10px;border-radius:10px;text-align:center;background:#f7f9fc}.kpi strong,.kpi span{display:block}.kpi strong{font-size:21px;color:#4f6ef7}.kpi span{font-size:11px;color:#8490a3;margin-top:5px}.kpi.cyan strong{color:#16a6a1}.kpi.purple strong{color:#8b5cf6}.kpi.orange strong{color:#e8684a}.chart-error{padding:12px;text-align:center;color:#b45309;background:#fff7ed;font-size:12px}@media(max-width:720px){.chart-canvas{height:270px}.kpi-grid{grid-template-columns:1fr 1fr}.kpi strong{font-size:17px}}
</style>

<script setup lang="ts">
import { computed } from 'vue'
import { Download } from '@element-plus/icons-vue'
import type { TaskResult } from '../types'
import ChartPanel from './ChartPanel.vue'

const props = withDefaults(defineProps<{ result: TaskResult; compact?: boolean }>(), { compact: false })
const charts = computed(() => props.result.charts || [])
const rows = computed(() => props.result.query_result || [])
const visibleRows = computed(() => props.compact ? rows.value.slice(0, 8) : rows.value)
const columns = computed(() => [...new Set(rows.value.flatMap(row => Object.keys(row || {})))])
const totalRows = computed(() => props.result.query_result_total_rows ?? rows.value.length)
const sourceTables = computed<string[]>(() => props.result.evidence?.source_tables || props.result.governance_result?.source_tables || [])
const qualityScore = computed(() => props.result.governance_result?.quality_score)
const evidence = computed(() => props.result.evidence)
const evidenceStatus = computed(() => evidence.value?.validation?.status || 'unverifiable')
const evidenceChecks = computed(() => evidence.value?.validation?.checks || [])
const evidenceFacts = computed(() => evidence.value?.facts || [])
const statusLabel = (status?: string) => ({ verified: '已验证', failed: '失败', unverifiable: '无法验证' }[status || ''] || status || '未知')
const statusType = (status?: string) => status === 'verified' ? 'success' : status === 'failed' ? 'danger' : 'warning'

function csvCell(value: unknown) {
  let text = value == null ? '' : typeof value === 'object' ? JSON.stringify(value) : String(value)
  if (/^[=+\-@]/.test(text)) text = `'${text}`
  return `"${text.replace(/"/g, '""')}"`
}

function downloadCsv() {
  const content = [columns.value.map(csvCell).join(','), ...rows.value.map(row => columns.value.map(column => csvCell(row[column])).join(','))].join('\r\n')
  const url = URL.createObjectURL(new Blob([`\ufeff${content}`], { type: 'text/csv;charset=utf-8' }))
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = `analysis-${props.result.request_id || Date.now()}.csv`
  anchor.click()
  URL.revokeObjectURL(url)
}
</script>

<template>
  <section v-if="charts.length || rows.length || result.demo_mode || sourceTables.length || evidence || result.demo_steps?.length || result.recovery_timeline?.length" class="result-visuals">
    <div v-if="result.demo_mode" class="demo-notice"><strong>缓存演示结果</strong><span>{{ result.source_notice }}</span></div>
    <div v-else-if="result.result_mode==='live_agent'" class="live-notice"><strong>实时 Agent 分析</strong><span>本结果来自本次工作流执行，不是缓存演示快照。</span></div>
    <div v-if="result.demo_steps?.length" class="demo-steps"><div v-for="(step,index) in result.demo_steps" :key="`${step.title}-${index}`"><b>{{ Number(index)+1 }}</b><span><strong>{{ step.title }}</strong><small>{{ step.detail }}</small></span><el-tag size="small" :type="statusType(step.status)">{{ statusLabel(step.status) }}</el-tag></div></div>
    <details v-if="result.context_changes?.length" class="demo-detail" open><summary>多轮条件迁移</summary><ol><li v-for="item in result.context_changes" :key="item.turn"><b>第 {{ item.turn }} 轮</b> {{ item.change }}</li></ol></details>
    <details v-if="result.recovery_timeline?.length" class="demo-detail" open><summary>故障恢复时间线</summary><div class="recovery-flow"><div v-for="(item,index) in result.recovery_timeline" :key="`${item.state}-${index}`"><b>{{ item.state }}</b><span>{{ item.owner || '无执行者' }}</span><small>{{ item.detail }}</small></div></div></details>
    <div v-if="sourceTables.length || qualityScore!=null" class="evidence-strip">
      <strong>证据来源</strong><span v-for="table in sourceTables" :key="table">{{ table }}</span><span v-if="qualityScore!=null">质量分 {{ Number(qualityScore).toFixed(2) }}</span><span v-if="result.request_id">任务 {{ String(result.request_id).slice(0,8) }}</span>
    </div>
    <details v-if="evidence" class="evidence-panel" :open="!compact">
      <summary><strong>证据链</strong><el-tag size="small" :type="statusType(evidenceStatus)">{{ statusLabel(evidenceStatus) }}</el-tag><span>{{ evidenceFacts.length }} 个事实</span></summary>
      <div class="evidence-meta"><span>指标目录 {{ evidence.metric_catalog_version || '未标注' }}</span><span>来源 {{ sourceTables.join(', ') || '未标注' }}</span></div>
      <div v-if="evidenceFacts.length" class="fact-list"><div v-for="fact in evidenceFacts" :key="fact.fact_id" class="fact-row"><code>{{ fact.fact_id }}</code><strong>{{ fact.label }}</strong><span>{{ fact.value }} {{ fact.unit || '' }}</span><small>{{ Object.entries(fact.dimensions || {}).map(([key,value]) => `${key}=${value}`).join(' · ') }}</small></div></div>
      <ul v-if="evidenceChecks.length" class="check-list"><li v-for="(check,index) in evidenceChecks" :key="`${check.code}-${index}`"><el-tag size="small" :type="statusType(check.status)">{{ statusLabel(check.status) }}</el-tag><code>{{ check.code }}</code><span>{{ check.message }}</span></li></ul>
      <div v-if="evidence.report_validation?.status" class="report-check"><strong>报告校验</strong><el-tag size="small" :type="statusType(evidence.report_validation.status)">{{ statusLabel(evidence.report_validation.status) }}</el-tag></div>
      <div v-if="evidence.limitations?.length" class="limitations"><strong>限制</strong><ul><li v-for="item in evidence.limitations" :key="item">{{ item }}</li></ul></div>
      <div v-if="evidence.sql" class="sql-evidence"><strong>执行 SQL</strong><pre>{{ evidence.sql }}</pre></div>
    </details>
    <div v-if="charts.length" class="visual-section">
      <div class="section-heading"><strong>数据可视化</strong><span>{{ charts.length }} 个图表</span></div>
      <div class="chart-grid"><ChartPanel v-for="(chart,index) in charts" :key="chart.id || index" :chart="chart" /></div>
    </div>
    <div v-if="rows.length" class="visual-section">
      <div class="section-heading"><strong>查询结果</strong><span>共 {{ totalRows }} 行<span v-if="result.query_result_truncated">，当前保留前 {{ rows.length }} 行</span></span><el-button v-if="!compact" link type="primary" :icon="Download" @click="downloadCsv">下载 CSV</el-button></div>
      <el-table :data="visibleRows" size="small" stripe border max-height="360"><el-table-column v-for="column in columns" :key="column" :prop="column" :label="column" min-width="130" show-overflow-tooltip /></el-table>
      <small v-if="compact && rows.length>visibleRows.length" class="more">对话中展示前 {{ visibleRows.length }} 行，完整结果可在任务详情查看。</small>
    </div>
  </section>
</template>

<style scoped>
.result-visuals{margin-top:14px}.demo-notice,.live-notice{display:flex;gap:9px;align-items:flex-start;padding:10px 12px;margin-bottom:12px;border:1px solid #f4d49a;border-radius:9px;background:#fff9ed;color:#8a5a08;font-size:11px}.demo-notice strong,.live-notice strong{flex:0 0 auto}.live-notice{border-color:#b9e3d5;background:#effbf6;color:#176b51}.demo-steps{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:10px 0 14px}.demo-steps>div{display:flex;align-items:center;gap:8px;padding:9px;border:1px solid #dfe5f0;border-radius:9px;background:#fff}.demo-steps>div>b{display:grid;place-items:center;width:22px;height:22px;border-radius:50%;background:#eef2ff;color:#4058c8}.demo-steps span{min-width:0;flex:1}.demo-steps strong,.demo-steps small{display:block}.demo-steps small{margin-top:3px;color:#7c8798}.demo-detail{margin:10px 0;padding:10px 12px;border:1px solid #dfe5f0;border-radius:9px;background:#fbfcff;font-size:11px}.demo-detail summary{font-weight:600;cursor:pointer}.demo-detail ol{margin:8px 0 0;padding-left:22px}.demo-detail li{margin:5px 0}.recovery-flow{display:grid;grid-template-columns:repeat(5,1fr);gap:7px;margin-top:9px}.recovery-flow>div{padding:8px;background:#fff;border-radius:7px}.recovery-flow b,.recovery-flow span,.recovery-flow small{display:block}.recovery-flow b{color:#4058c8}.recovery-flow span{margin:3px 0;color:#596579}.recovery-flow small{color:#8a94a5}.evidence-strip{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin-bottom:12px;font-size:10px;color:#657086}.evidence-strip strong{font-size:11px;color:#344056}.evidence-strip span{padding:3px 7px;background:#edf1f8;border-radius:9px}.visual-section+.visual-section{margin-top:16px}.section-heading{display:flex;align-items:center;gap:10px;margin-bottom:9px}.section-heading strong{font-size:13px}.section-heading span{font-size:10px;color:#8c96a7}.section-heading .el-button{margin-left:auto}.chart-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.chart-grid>*:only-child{grid-column:1/-1}.more{display:block;margin-top:7px;color:#9099a8}@media(max-width:1000px){.chart-grid{grid-template-columns:1fr}.demo-steps{grid-template-columns:1fr}.recovery-flow{grid-template-columns:1fr 1fr}}
.evidence-panel{margin:10px 0 14px;padding:10px 12px;border:1px solid #dce3ef;border-radius:10px;background:#fbfcff;font-size:11px}.evidence-panel summary{display:flex;align-items:center;gap:8px;cursor:pointer;color:#344056}.evidence-panel summary>span{color:#7b8699}.evidence-meta{display:flex;gap:14px;margin:10px 0;color:#657086}.fact-list{display:grid;gap:6px}.fact-row{display:grid;grid-template-columns:40px minmax(100px,1fr) minmax(100px,auto) minmax(120px,1fr);gap:8px;align-items:center;padding:6px 8px;background:#fff;border-radius:6px}.fact-row small{color:#7b8699}.check-list{display:grid;gap:6px;padding:0;margin:10px 0;list-style:none}.check-list li{display:flex;align-items:center;gap:7px}.report-check{display:flex;align-items:center;gap:8px;margin-top:10px}.limitations{margin-top:10px;color:#8a5a08}.limitations ul{margin:4px 0 0;padding-left:18px}.sql-evidence{margin-top:10px}.sql-evidence pre{overflow:auto;max-height:180px;padding:8px;border-radius:6px;background:#182030;color:#e7edf8;white-space:pre-wrap}@media(max-width:700px){.fact-row{grid-template-columns:38px 1fr}.fact-row small{grid-column:1/-1}}
</style>

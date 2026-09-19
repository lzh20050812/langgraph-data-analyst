<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { Coin, View } from '@element-plus/icons-vue'
import { api, errorMessage } from '../api/client'

interface Field { name:string; type:string; nullable:boolean; business_term:string; aliases:string[] }
interface TableInfo { table_name:string; row_count:number; column_count:number; fields:Field[] }
const tables = ref<TableInfo[]>([])
const selected = ref<TableInfo | null>(null)
const preview = ref<{columns:string[];rows:Record<string,any>[]} | null>(null)
const previewVisible = ref(false)
const loading = ref(true)

async function load() { try { const data=await api<{tables:TableInfo[]}>('/data/catalog'); tables.value=data.tables; selected.value=tables.value[0]||null } catch(error:unknown){ElMessage.error(errorMessage(error,'数据目录加载失败'))} finally { loading.value=false } }
async function showPreview(table:TableInfo) { selected.value=table; try { preview.value=await api(`/data/tables/${table.table_name}/preview?limit=20`); previewVisible.value=true } catch(error:unknown){ElMessage.error(errorMessage(error,'预览失败'))} }
onMounted(load)
</script>

<template><div class="page"><div class="page-heading"><div><h2>数据中心</h2><p>浏览允许访问的数据表、字段语义和受控样例数据</p></div><el-tag type="success" effect="light">只读数据目录</el-tag></div>
<el-skeleton v-if="loading" :rows="6" animated />
<template v-else><div class="catalog-grid"><button v-for="table in tables" :key="table.table_name" :class="['table-card','surface',{active:selected?.table_name===table.table_name}]" @click="selected=table"><div class="table-icon"><el-icon><Coin /></el-icon></div><div><strong>{{ table.table_name }}</strong><span>{{ table.row_count.toLocaleString() }} 行 · {{ table.column_count }} 字段</span></div><el-button link type="primary" :icon="View" @click.stop="showPreview(table)">预览</el-button></button></div>
<div v-if="selected" class="surface schema-panel"><div class="table-header"><div><h3>{{ selected.table_name }} 字段字典</h3><small class="muted">业务术语来自 Schema Agent 元数据</small></div><el-button type="primary" plain :icon="View" @click="showPreview(selected)">查看数据预览</el-button></div><el-table :data="selected.fields" stripe><el-table-column prop="name" label="字段" min-width="180"><template #default="scope"><code>{{ scope.row.name }}</code></template></el-table-column><el-table-column prop="type" label="类型" width="150"/><el-table-column label="必填" width="90"><template #default="scope"><el-tag size="small" :type="scope.row.nullable?'info':'success'">{{ scope.row.nullable?'可空':'必填' }}</el-tag></template></el-table-column><el-table-column prop="business_term" label="业务含义" min-width="260"/><el-table-column label="别名" min-width="260"><template #default="scope"><el-tag v-for="alias in scope.row.aliases.slice(0,4)" :key="alias" size="small" effect="plain" class="alias">{{ alias }}</el-tag></template></el-table-column></el-table></div></template>
<el-drawer v-model="previewVisible" :title="`${selected?.table_name || ''} · 前 20 行安全预览`" size="75%"><el-table v-if="preview" :data="preview.rows" border height="calc(100vh - 150px)"><el-table-column v-for="column in preview.columns" :key="column" :prop="column" :label="column" min-width="150" show-overflow-tooltip /></el-table></el-drawer></div></template>

<style scoped>.catalog-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin-bottom:18px}.table-card{padding:16px;border:1px solid var(--border);display:grid;grid-template-columns:40px 1fr auto;align-items:center;gap:10px;text-align:left;cursor:pointer}.table-card.active{border-color:#7186f5;box-shadow:0 0 0 2px rgba(79,110,247,.1)}.table-card strong,.table-card span{display:block}.table-card strong{font-size:14px}.table-card span{color:#8a94a6;font-size:11px;margin-top:4px}.table-icon{width:38px;height:38px;border-radius:10px;background:#eef2ff;color:#4f6ef7;display:grid;place-items:center;font-size:20px}.schema-panel{overflow:hidden}.table-header small{display:block;margin-top:4px}.alias{margin:2px 4px 2px 0}@media(max-width:1200px){.catalog-grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:700px){.catalog-grid{grid-template-columns:1fr}.table-header{align-items:flex-start;gap:12px;flex-direction:column}}</style>

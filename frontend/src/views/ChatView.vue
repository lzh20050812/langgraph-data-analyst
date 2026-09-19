<script setup lang="ts">
import { nextTick, onMounted, onUnmounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { ChatDotRound, Plus, Promotion, Loading, Edit, Delete } from '@element-plus/icons-vue'
import { api, apiUrl, errorMessage } from '../api/client'
import type { AnalysisRequest, Conversation, Message, Task } from '../types'
import ResultVisuals from '../components/ResultVisuals.vue'
import StructuredReport from '../components/StructuredReport.vue'

interface DemoScenario { id: string; title: string; description: string; query: string; kind: string; steps: string[]; mode: 'cached_demo' }
interface MessageSubmission {
  task_id?: string
  status: 'queued' | 'waiting_clarification' | 'unsupported'
  detail?: string
  analysis_request: AnalysisRequest
  context_changes: Record<string, any>[]
}

const conversations = ref<Conversation[]>([])
const active = ref<Conversation | null>(null)
const messages = ref<Message[]>([])
const input = ref('')
const sending = ref(false)
const creating = ref(false)
const progress = ref<{ node: string; state: string; duration?: number }[]>([])
const taskResults = ref<Record<string, Task>>({})
const demoScenarios = ref<DemoScenario[]>([])
const runningDemo = ref('')
const messageArea = ref<HTMLElement>()
const activeStreams = new Set<EventSource>()
let disposed = false
const examples = ['查询中国区 GMV 最高的 10 个客户', '各品类营收与退货率', '预测未来 6 个月销售趋势', '分析客户价值并预测流失风险']

async function loadConversations(selectFirst = true) {
  const data = await api<{ conversations: Conversation[] }>('/conversations')
  conversations.value = data.conversations
  if (selectFirst && !active.value && conversations.value.length) await selectConversation(conversations.value[0])
}

async function createConversation() {
  if (creating.value) return
  creating.value = true
  try {
    const conversation = await api<Conversation>('/conversations', {
      method: 'POST', body: JSON.stringify({ title: '新分析对话' }),
    })
    conversations.value.unshift(conversation)
    await selectConversation(conversation)
  } finally { creating.value = false }
}

async function renameConversation(conversation: Conversation) {
  try {
    const { value } = await ElMessageBox.prompt('输入新的对话名称', '重命名对话', {
      inputValue: conversation.title, inputPattern: /\S+/, inputErrorMessage: '名称不能为空', confirmButtonText: '保存', cancelButtonText: '取消',
    })
    const updated = await api<Conversation>(`/conversations/${conversation.conversation_id}`, {
      method: 'PATCH', body: JSON.stringify({ title: value.trim() }),
    })
    conversations.value = conversations.value.map(item => item.conversation_id === updated.conversation_id ? updated : item)
    if (active.value?.conversation_id === updated.conversation_id) active.value = updated
    ElMessage.success('对话已重命名')
  } catch (error: unknown) {
    if (error !== 'cancel' && error !== 'close') ElMessage.error(errorMessage(error, '重命名失败'))
  }
}

async function deleteConversation(conversation: Conversation) {
  try {
    await ElMessageBox.confirm(`将永久删除“${conversation.title}”及其对话消息，任务记录仍保留。`, '删除对话', {
      type: 'warning', confirmButtonText: '删除', cancelButtonText: '取消', confirmButtonClass: 'el-button--danger',
    })
    await api(`/conversations/${conversation.conversation_id}`, { method: 'DELETE' })
    const wasActive = active.value?.conversation_id === conversation.conversation_id
    conversations.value = conversations.value.filter(item => item.conversation_id !== conversation.conversation_id)
    if (wasActive) {
      active.value = null; messages.value = []
      if (conversations.value.length) await selectConversation(conversations.value[0])
      else await createConversation()
    }
    ElMessage.success('对话已删除')
  } catch (error: unknown) {
    if (error !== 'cancel' && error !== 'close') ElMessage.error(errorMessage(error, '删除失败'))
  }
}

async function loadDemoScenarios() {
  const data = await api<{ enabled: boolean; scenarios: DemoScenario[] }>('/demo/scenarios')
  demoScenarios.value = data.enabled ? data.scenarios : []
}

async function runDemo(scenario: DemoScenario) {
  if (runningDemo.value || sending.value) return
  if (!active.value) await createConversation()
  runningDemo.value = scenario.id
  try {
    const receipt = await api<{ task_id: string }>(`/conversations/${active.value!.conversation_id}/demo/${scenario.id}`, { method: 'POST' })
    const task = await api<Task>(`/tasks/${receipt.task_id}`)
    taskResults.value = { ...taskResults.value, [receipt.task_id]: task }
    await selectConversation(active.value!)
    await loadConversations(false)
    ElMessage.success('缓存演示结果已载入')
  } catch (error: unknown) {
    ElMessage.error(errorMessage(error, '演示场景加载失败'))
  } finally { runningDemo.value = '' }
}

async function selectConversation(conversation: Conversation) {
  const data = await api<Conversation & { messages: Message[] }>(`/conversations/${conversation.conversation_id}`)
  active.value = data
  messages.value = data.messages
  progress.value = []
  void hydrateTaskResults(data.messages)
  await scrollBottom()
}

async function hydrateTaskResults(items: Message[]) {
  const taskIds = [...new Set(items.map(item => item.task_id).filter((id): id is string => Boolean(id)))].slice(-12)
  await Promise.all(taskIds.map(async taskId => {
    if (taskResults.value[taskId]) return
    try {
      const task = await api<Task>(`/tasks/${taskId}`)
      taskResults.value = { ...taskResults.value, [taskId]: task }
    } catch (_) { /* Removed or inaccessible historical tasks have no visuals. */ }
  }))
}

async function send() {
  const content = input.value.trim()
  if (!content || sending.value) return
  if (!active.value) await createConversation()
  const conversationId = active.value!.conversation_id
  input.value = ''
  sending.value = true
  progress.value = []
  try {
    const submission = await api<MessageSubmission>(`/conversations/${conversationId}/messages`, {
      method: 'POST', body: JSON.stringify({ content }),
    })
    await selectConversation(active.value!)
    if (submission.status !== 'queued') {
      if (submission.status === 'waiting_clarification') ElMessage.info(submission.detail || '请补充分析口径')
      else ElMessage.warning(submission.detail || '当前数据不支持该请求')
      await loadConversations(false)
      return
    }
    const taskId = submission.task_id
    if (!taskId) throw new Error('服务端未返回任务编号')
    followEvents(taskId)
    const completedTask = await pollTask(taskId)
    if (!completedTask) return
    taskResults.value = { ...taskResults.value, [taskId]: completedTask }
    await selectConversation(active.value!)
    await loadConversations(false)
  } catch (error: unknown) {
    ElMessage.error(errorMessage(error, '消息发送失败'))
  } finally { sending.value = false }
}

function followEvents(taskId: string) {
  const stream = new EventSource(apiUrl(`/tasks/${taskId}/events`), { withCredentials: true })
  activeStreams.add(stream)
  const close = () => { stream.close(); activeStreams.delete(stream) }
  const nodes = new Map<string, { node: string; state: string; duration?: number }>()
  const update = (event: MessageEvent) => {
    try {
      const payload = JSON.parse(event.data)
      if (!payload.node) return
      nodes.set(payload.node, {
        node: payload.node,
        state: payload.type === 'node_completed' ? 'completed' : payload.type === 'node_failed' ? 'failed' : 'running',
        duration: payload.duration_ms,
      })
      progress.value = [...nodes.values()]
    } catch (_) { /* ignore malformed progress */ }
  }
  ;['node_started','node_completed','node_failed'].forEach(name => stream.addEventListener(name, update as EventListener))
  ;['task_completed','task_failed','task_cancelled'].forEach(name => stream.addEventListener(name, close))
  stream.onerror = close
}

async function pollTask(taskId: string) {
  while (!disposed) {
    const task = await api<Task>(`/tasks/${taskId}`)
    if (['completed','failed','cancelled','interrupted'].includes(task.status)) {
      if (task.status !== 'completed') throw new Error(task.error || `任务${task.status}`)
      return task
    }
    await new Promise(resolve => setTimeout(resolve, 600))
  }
  return null
}

async function scrollBottom() { await nextTick(); if (messageArea.value) messageArea.value.scrollTop = messageArea.value.scrollHeight }
function useExample(text: string) { input.value = text }
function prepareConditionEdit(mode: 'modify' | 'new') {
  input.value = mode === 'new' ? '新分析 ' : '修改条件：'
}
function nodeLabel(node: string) { return ({planner:'需求规划',schema_agent:'Schema 检索',sql_agent:'SQL 执行',governance_agent:'数据治理',analysis_agent:'数据分析',prediction_agent:'预测建模',report_agent:'报告生成',chart_renderer:'图表渲染'} as Record<string,string>)[node] || node }
function contextSummary(request?: AnalysisRequest | null) {
  if (!request) return []
  const items = [
    ...request.metrics.map(value => `指标: ${value}`),
    ...request.dimensions.map(value => `维度: ${value}`),
    ...request.time_scope.years.map(value => `年份: ${value}`),
    ...Object.entries(request.filters).flatMap(([key, values]) => values.map(value => `${key}: ${value}`)),
  ]
  if (request.limit) items.push(`数量: ${request.limit}`)
  return items
}

onMounted(async () => {
  disposed = false
  try {
    await Promise.all([loadConversations(), loadDemoScenarios()])
    if (!conversations.value.length) await createConversation()
  } catch (error: unknown) {
    ElMessage.error(errorMessage(error, '对话列表加载失败'))
  }
})
onUnmounted(() => {
  disposed = true
  activeStreams.forEach(stream => stream.close())
  activeStreams.clear()
})
</script>

<template>
  <div class="chat-page">
    <aside class="conversation-panel surface">
      <el-button type="primary" :icon="Plus" :loading="creating" class="new-chat" @click="createConversation">新建对话</el-button>
      <div class="conversation-list">
        <div v-for="item in conversations" :key="item.conversation_id" :class="['conversation-row',{active:item.conversation_id===active?.conversation_id}]">
          <button class="conversation-main" @click="selectConversation(item)"><el-icon><ChatDotRound /></el-icon><span><strong>{{ item.title }}</strong><small>{{ new Date(item.updated_at*1000).toLocaleString() }}</small></span></button>
          <div class="conversation-actions"><button title="重命名" @click="renameConversation(item)"><el-icon><Edit /></el-icon></button><button title="删除" @click="deleteConversation(item)"><el-icon><Delete /></el-icon></button></div>
        </div>
      </div>
    </aside>
    <section class="chat-workspace surface">
      <div class="chat-title"><div><strong>{{ active?.title || '智能分析助手' }}</strong><span><i class="status-dot"></i>多智能体服务在线</span></div></div>
      <div v-if="active?.analysis_context || active?.pending_clarification" class="analysis-context" :class="{pending:Boolean(active?.pending_clarification)}">
        <div><strong>{{ active?.pending_clarification?'待澄清条件':'本次分析条件' }}</strong><small>口径 {{ (active?.pending_clarification||active?.analysis_context)?.metric_catalog_version }}</small></div>
        <span v-for="item in contextSummary(active?.pending_clarification||active?.analysis_context)" :key="item">{{ item }}</span>
        <div class="context-actions"><button @click="prepareConditionEdit('modify')">修改条件</button><button @click="prepareConditionEdit('new')">新分析</button></div>
      </div>
      <div ref="messageArea" class="messages">
        <div v-if="!messages.length" class="chat-empty">
          <div class="ai-orb"><el-icon><ChatDotRound /></el-icon></div><h2>你好，我是你的 AI 数据分析师</h2><p>描述业务问题，我会自动规划查询、分析、预测并生成可追溯报告。</p>
          <div v-if="demoScenarios.length" class="demo-zone"><div class="demo-heading"><strong>答辩演示</strong><span>固定数据集缓存 · 不调用 LLM</span></div><div class="demo-cards"><button v-for="scenario in demoScenarios" :key="scenario.id" :disabled="Boolean(runningDemo)" @click="runDemo(scenario)"><strong>{{ scenario.title }}</strong><span>{{ scenario.description }}</span><small>{{ runningDemo===scenario.id?'载入中…':scenario.kind==='failure_recovery'?'演示恢复链路':'一键展示' }}</small></button></div></div>
          <div class="examples"><button v-for="example in examples" :key="example" @click="useExample(example)">{{ example }}</button></div>
        </div>
        <div v-for="message in messages" :key="message.message_id" :class="['message',message.role]">
          <div class="avatar">{{ message.role==='user'?'我':message.role==='assistant'?'AI':'!' }}</div>
          <div :class="['bubble',{'with-visuals':message.task_id&&taskResults[message.task_id]?.result}]"><div class="message-role">{{ message.role==='user'?'你':message.role==='assistant'?'AI 数据分析师':'系统' }}</div><StructuredReport v-if="message.role==='assistant'" :content="message.content"/><div v-else class="message-content">{{ message.content }}</div><ResultVisuals v-if="message.task_id&&taskResults[message.task_id]?.result" :result="taskResults[message.task_id].result!" compact/><router-link v-if="message.task_id" :to="`/tasks?task=${message.task_id}`">查看任务详情 →</router-link></div>
        </div>
        <div v-if="sending" class="message assistant"><div class="avatar">AI</div><div class="bubble progress-bubble"><div class="message-role"><el-icon class="spin"><Loading /></el-icon> 多智能体正在分析</div><div class="progress-nodes"><span v-for="item in progress" :key="item.node" :class="item.state">{{ item.state==='completed'?'✓':'·' }} {{ nodeLabel(item.node) }}</span><small v-if="!progress.length">正在创建任务...</small></div></div></div>
      </div>
      <div class="composer"><div class="composer-box"><el-input v-model="input" type="textarea" :autosize="{minRows:1,maxRows:4}" resize="none" placeholder="输入分析问题，Enter 发送，Shift+Enter 换行" @keydown.enter.exact.prevent="send" /><el-button aria-label="发送分析问题" circle type="primary" :icon="Promotion" :loading="sending" @click="send" /></div><small>AI 输出应结合数据证据进行判断，重要决策请人工复核。</small></div>
    </section>
  </div>
</template>

<style scoped>
.chat-page{height:calc(100vh - 76px);padding:18px;display:grid;grid-template-columns:250px 1fr;gap:14px}.conversation-panel{overflow:hidden;display:flex;flex-direction:column}.new-chat{margin:15px}.conversation-list{overflow:auto;padding:0 9px 12px}.conversation-row{display:flex;align-items:center;border-radius:9px;color:#596579}.conversation-row:hover,.conversation-row.active{background:#eef2ff;color:#334ed0}.conversation-main{min-width:0;flex:1;border:0;background:transparent;padding:11px 4px 11px 11px;display:flex;gap:9px;text-align:left;color:inherit;cursor:pointer}.conversation-main span,.conversation-main strong,.conversation-main small{display:block;min-width:0}.conversation-main span{overflow:hidden}.conversation-main strong{font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.conversation-main small{font-size:10px;margin-top:4px;color:#9aa4b5}.conversation-actions{display:none;padding-right:5px}.conversation-row:hover .conversation-actions,.conversation-row.active .conversation-actions{display:flex}.conversation-actions button{border:0;background:transparent;color:#7b8697;padding:5px;cursor:pointer}.conversation-actions button:hover{color:#334ed0}.chat-workspace{display:flex;flex-direction:column;overflow:hidden}.chat-title{height:60px;padding:0 20px;border-bottom:1px solid var(--border);display:flex;align-items:center}.chat-title strong,.chat-title span{display:block}.chat-title span{font-size:11px;color:#8b95a7;margin-top:3px}.analysis-context{padding:8px 20px;border-bottom:1px solid #dce3fb;background:#f7f9ff;display:flex;align-items:center;gap:7px;flex-wrap:wrap}.analysis-context.pending{background:#fff8e8;border-color:#f2d69b}.analysis-context>div{margin-right:5px}.analysis-context strong,.analysis-context small{display:block}.analysis-context strong{font-size:12px;color:#334ed0}.analysis-context small{font-size:9px;color:#8b95a7}.analysis-context span{font-size:10px;padding:3px 7px;border-radius:8px;background:#fff;border:1px solid #dce3fb;color:#536076}.messages{flex:1;overflow:auto;padding:24px 8%}.chat-empty{text-align:center;padding:5vh 0;color:#738096}.chat-empty h2{color:#263248;margin:16px 0 5px}.ai-orb{margin:auto;width:58px;height:58px;border-radius:18px;background:linear-gradient(135deg,#4f6ef7,#36cfc9);color:#fff;display:grid;place-items:center;font-size:28px;box-shadow:0 12px 28px rgba(79,110,247,.24)}.demo-zone{max-width:760px;margin:26px auto 8px;text-align:left;padding:15px;border:1px solid #dce3fb;border-radius:13px;background:linear-gradient(135deg,#f7f9ff,#f0fbfa)}.demo-heading{display:flex;align-items:center;gap:9px;margin-bottom:10px}.demo-heading strong{color:#344ed0}.demo-heading span{font-size:10px;padding:3px 7px;background:#fff4d8;color:#8a5a08;border-radius:8px}.demo-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}.demo-cards button{border:1px solid #e1e6f4;background:#fff;border-radius:10px;padding:11px;text-align:left;cursor:pointer}.demo-cards button:disabled{opacity:.6;cursor:wait}.demo-cards strong,.demo-cards span,.demo-cards small{display:block}.demo-cards strong{font-size:12px;color:#2e3950}.demo-cards span{font-size:10px;line-height:1.5;margin:5px 0;color:#778397}.demo-cards small{color:#4f6ef7}.examples{display:grid;grid-template-columns:1fr 1fr;gap:10px;max-width:620px;margin:18px auto}.examples button{border:1px solid #e2e7f0;background:#fff;border-radius:10px;padding:13px;color:#536076;cursor:pointer;text-align:left}.examples button:hover{border-color:#8093f8;color:#3f57d6}.message{display:flex;gap:12px;margin:20px 0}.message.user{flex-direction:row-reverse}.avatar{width:32px;height:32px;border-radius:10px;background:#15213a;color:#fff;display:grid;place-items:center;font-size:11px;flex:0 0 auto}.message.user .avatar{background:#4f6ef7}.bubble{max-width:78%;padding:13px 16px;background:#f4f6fa;border-radius:4px 14px 14px 14px}.bubble.with-visuals{width:96%;max-width:96%}.message.user .bubble{background:#eef2ff;border-radius:14px 4px 14px 14px}.message-role{font-weight:600;font-size:12px;margin-bottom:5px}.message-content{white-space:pre-wrap;line-height:1.75;font-size:14px}.bubble a{display:block;margin-top:8px;color:#4f6ef7;font-size:12px}.progress-bubble{min-width:360px}.progress-nodes{display:flex;flex-wrap:wrap;gap:6px}.progress-nodes span{font-size:11px;padding:4px 8px;border-radius:10px;background:#e9edf5}.progress-nodes span.completed{background:#dcfce7;color:#166534}.progress-nodes span.failed{background:#fee2e2;color:#991b1b}.spin{animation:spin 1s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}.composer{padding:14px 7% 12px;border-top:1px solid var(--border)}.composer-box{display:flex;align-items:flex-end;gap:10px;padding:9px 10px 9px 15px;border:1px solid #dce2ec;border-radius:13px;box-shadow:0 4px 18px rgba(15,23,42,.05)}.composer :deep(.el-textarea__inner){box-shadow:none;padding:7px 0}.composer>small{display:block;text-align:center;color:#a0a8b5;margin-top:7px;font-size:10px}
.analysis-context .context-actions{margin-left:auto;display:flex;gap:5px}.context-actions button{border:1px solid #bfcaf2;border-radius:7px;background:#fff;color:#4058c8;padding:4px 8px;font-size:10px;cursor:pointer}.context-actions button:hover{background:#eef2ff}
@media (max-width:900px){.chat-page{grid-template-columns:1fr;padding:10px}.conversation-panel{display:none}.messages{padding:18px 4%}.composer{padding:12px 4%}.examples,.demo-cards{grid-template-columns:1fr}.progress-bubble{min-width:0}.bubble{max-width:88%}}
</style>

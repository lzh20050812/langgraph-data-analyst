<script setup lang="ts">
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'
import {
  ChatDotRound, Coin, List, Monitor, Document, UserFilled,
  DataAnalysis, SwitchButton,
} from '@element-plus/icons-vue'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()
const title = computed(() => String(route.meta.title || '智能数据分析系统'))

async function logout() {
  await auth.logout()
  router.replace('/login')
}
</script>

<template>
  <div class="portal-shell">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark"><el-icon><DataAnalysis /></el-icon></div>
        <div><strong>AI Analyst</strong><span>智能数据分析系统</span></div>
      </div>
      <el-menu :default-active="route.path" router class="portal-menu">
        <el-menu-item index="/chat"><el-icon><ChatDotRound /></el-icon><span>对话分析</span></el-menu-item>
        <el-menu-item index="/data"><el-icon><Coin /></el-icon><span>数据中心</span></el-menu-item>
        <el-menu-item index="/tasks"><el-icon><List /></el-icon><span>任务中心</span></el-menu-item>
        <el-menu-item index="/operations"><el-icon><Monitor /></el-icon><span>运行监控</span></el-menu-item>
        <el-menu-item v-if="auth.isAdmin" index="/audit"><el-icon><Document /></el-icon><span>日志中心</span></el-menu-item>
        <el-menu-item v-if="auth.isAdmin" index="/users"><el-icon><UserFilled /></el-icon><span>用户管理</span></el-menu-item>
      </el-menu>
      <div class="sidebar-foot">
        <span class="status-dot"></span>系统在线
        <small>LangGraph · FastAPI</small>
      </div>
    </aside>
    <section class="workspace">
      <header class="topbar">
        <div><h1>{{ title }}</h1><span>企业运营数据智能分析平台</span></div>
        <el-dropdown>
          <div class="user-chip">
            <el-avatar :size="34">{{ (auth.user?.display_name || 'U').slice(0,1) }}</el-avatar>
            <div><strong>{{ auth.user?.display_name || auth.identity?.subject }}</strong><small>{{ auth.isAdmin ? '管理员' : '分析员' }}</small></div>
          </div>
          <template #dropdown>
            <el-dropdown-menu><el-dropdown-item @click="logout"><el-icon><SwitchButton /></el-icon>退出登录</el-dropdown-item></el-dropdown-menu>
          </template>
        </el-dropdown>
      </header>
      <main class="content"><router-view /></main>
    </section>
  </div>
</template>

<style scoped>
.portal-shell{display:flex;min-height:100vh}.sidebar{position:fixed;inset:0 auto 0 0;width:232px;background:linear-gradient(180deg,#111827,#0f172a);color:#fff;display:flex;flex-direction:column;z-index:10}.brand{height:76px;display:flex;align-items:center;gap:12px;padding:0 20px;border-bottom:1px solid rgba(255,255,255,.08)}.brand-mark{width:38px;height:38px;border-radius:11px;background:linear-gradient(135deg,#657ef8,#36cfc9);display:grid;place-items:center;font-size:21px}.brand strong,.brand span{display:block}.brand strong{font-size:17px}.brand span{font-size:11px;color:#94a3b8;margin-top:2px}.portal-menu{border:0;background:transparent;padding:14px 10px;flex:1}.portal-menu :deep(.el-menu-item){color:#aeb8ca;border-radius:8px;margin:3px 0}.portal-menu :deep(.el-menu-item:hover){background:#1e293b;color:#fff}.portal-menu :deep(.el-menu-item.is-active){background:linear-gradient(90deg,rgba(79,110,247,.3),rgba(79,110,247,.08));color:#fff}.sidebar-foot{padding:18px 20px;border-top:1px solid rgba(255,255,255,.08);font-size:12px}.sidebar-foot small{display:block;color:#64748b;margin-top:4px}.workspace{margin-left:232px;min-height:100vh;width:calc(100% - 232px)}.topbar{height:76px;padding:0 28px;background:#fff;border-bottom:1px solid #e7ebf2;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:8}.topbar h1{margin:0;font-size:19px}.topbar>div>span{font-size:11px;color:#94a3b8}.user-chip{display:flex;align-items:center;gap:10px;cursor:pointer}.user-chip strong,.user-chip small{display:block}.user-chip strong{font-size:13px}.user-chip small{font-size:11px;color:#94a3b8;margin-top:2px}.content{min-height:calc(100vh - 76px)}
@media (max-width:900px){.sidebar{width:72px}.brand{padding:0 17px}.brand>div:last-child,.portal-menu span,.sidebar-foot{display:none}.portal-menu{padding:14px 8px}.portal-menu :deep(.el-menu-item){justify-content:center;padding:0!important}.portal-menu :deep(.el-menu-item .el-icon){margin:0}.workspace{margin-left:72px;width:calc(100% - 72px)}.topbar{padding:0 16px}.topbar>div>span,.user-chip>div{display:none}}
</style>

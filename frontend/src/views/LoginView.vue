<script setup lang="ts">
import { reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { DataAnalysis, Lock, User } from '@element-plus/icons-vue'
import { useAuthStore } from '../stores/auth'
import { errorMessage } from '../api/client'

const auth = useAuthStore()
const router = useRouter()
const route = useRoute()
const loading = ref(false)
const form = reactive({ username: 'admin', password: '' })

async function submit() {
  loading.value = true
  try {
    await auth.login(form.username, form.password)
    ElMessage.success('登录成功')
    router.replace(String(route.query.redirect || '/chat'))
  } catch (error: unknown) {
    ElMessage.error(errorMessage(error, '登录失败'))
  } finally { loading.value = false }
}
</script>

<template>
  <div class="login-page">
    <section class="login-hero">
      <div class="hero-content">
        <div class="hero-badge"><el-icon><DataAnalysis /></el-icon> AI DATA ANALYST</div>
        <h1>让企业数据<br><span>真正产生洞察</span></h1>
        <p>基于 LangGraph 的多智能体协同分析平台，从自然语言问题到可信 SQL、预测模型与经营报告。</p>
        <div class="hero-points"><span>✓ 证据约束</span><span>✓ 全链路追踪</span><span>✓ 企业级治理</span></div>
      </div>
    </section>
    <section class="login-panel">
      <div class="login-card">
        <div class="mobile-logo"><el-icon><DataAnalysis /></el-icon></div>
        <h2>欢迎登录</h2><p>进入智能数据分析工作台</p>
        <el-form @submit.prevent="submit" size="large">
          <el-form-item><el-input v-model="form.username" autocomplete="username" placeholder="用户名" :prefix-icon="User" /></el-form-item>
          <el-form-item><el-input v-model="form.password" type="password" autocomplete="current-password" show-password placeholder="密码" :prefix-icon="Lock" @keyup.enter="submit" /></el-form-item>
          <el-button type="primary" :loading="loading" class="login-btn" @click="submit">登录系统</el-button>
        </el-form>
        <div class="demo-tip"><strong>本地演示账号</strong><span>admin / Admin123!</span><small>共享部署前请在 .env 中修改初始密码</small></div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.login-page{min-height:100vh;display:grid;grid-template-columns:1.15fr .85fr;background:#fff}.login-hero{position:relative;overflow:hidden;background:radial-gradient(circle at 15% 20%,rgba(79,110,247,.35),transparent 33%),radial-gradient(circle at 85% 80%,rgba(54,207,201,.18),transparent 34%),linear-gradient(145deg,#0b1220,#14213d);color:#fff;display:flex;align-items:center;padding:10%}.login-hero:after{content:'';position:absolute;inset:0;background-image:linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px);background-size:42px 42px}.hero-content{position:relative;z-index:1;max-width:610px}.hero-badge{display:inline-flex;align-items:center;gap:8px;padding:7px 12px;border:1px solid rgba(255,255,255,.16);border-radius:20px;color:#a5b4fc;font-size:12px;letter-spacing:1px}.hero-content h1{font-size:52px;line-height:1.18;margin:28px 0 20px;letter-spacing:-2px}.hero-content h1 span{background:linear-gradient(90deg,#8ea2ff,#67e8d4);-webkit-background-clip:text;color:transparent}.hero-content p{font-size:16px;line-height:1.9;color:#aeb8ca;max-width:570px}.hero-points{display:flex;gap:25px;margin-top:34px;color:#dbeafe;font-size:13px}.login-panel{display:grid;place-items:center;padding:40px}.login-card{width:390px}.mobile-logo{width:48px;height:48px;border-radius:13px;background:#eef2ff;color:#4f6ef7;display:grid;place-items:center;font-size:25px}.login-card h2{font-size:28px;margin:24px 0 6px}.login-card>p{color:#8a94a6;margin:0 0 30px}.login-btn{width:100%;height:46px;font-weight:600;background:#4f6ef7}.demo-tip{margin-top:26px;padding:14px 16px;border-radius:10px;background:#f7f9fd;border:1px solid #e7ebf3;font-size:12px}.demo-tip strong,.demo-tip span,.demo-tip small{display:block}.demo-tip span{font-family:monospace;margin:5px 0;color:#4f6ef7}.demo-tip small{color:#98a2b3}
@media (max-width:780px){.login-page{display:block;background:linear-gradient(180deg,#eef2ff,#fff 35%)}.login-hero{display:none}.login-panel{min-height:100vh;padding:28px 20px}.login-card{width:min(390px,100%)}.login-card h2{font-size:25px}}
</style>

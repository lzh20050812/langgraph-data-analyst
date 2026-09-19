import { createRouter, createWebHashHistory } from 'vue-router'
import { useAuthStore } from '../stores/auth'

const AppLayout = () => import('../layouts/AppLayout.vue')
const LoginView = () => import('../views/LoginView.vue')
const ChatView = () => import('../views/ChatView.vue')
const DataView = () => import('../views/DataView.vue')
const TasksView = () => import('../views/TasksView.vue')
const OperationsView = () => import('../views/OperationsView.vue')
const AuditView = () => import('../views/AuditView.vue')
const UsersView = () => import('../views/UsersView.vue')

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/login', name: 'login', component: LoginView, meta: { public: true } },
    {
      path: '/', component: AppLayout,
      children: [
        { path: '', redirect: '/chat' },
        { path: 'chat', component: ChatView, meta: { title: '对话分析' } },
        { path: 'data', component: DataView, meta: { title: '数据中心' } },
        { path: 'tasks', component: TasksView, meta: { title: '任务中心' } },
        { path: 'operations', component: OperationsView, meta: { title: '运行监控' } },
        { path: 'audit', component: AuditView, meta: { title: '日志中心', admin: true } },
        { path: 'users', component: UsersView, meta: { title: '用户管理', admin: true } },
      ],
    },
    { path: '/:pathMatch(.*)*', redirect: '/chat' },
  ],
})

router.beforeEach(async (to) => {
  if (to.meta.public) return true
  const auth = useAuthStore()
  if (!(await auth.ensure())) return { path: '/login', query: { redirect: to.fullPath } }
  if (to.meta.admin && !auth.isAdmin) return '/chat'
  return true
})

router.afterEach((to) => {
  const title = String(to.meta.title || (to.name === 'login' ? '登录' : '智能分析'))
  document.title = `${title} · AI Data Analyst`
})

window.addEventListener('portal:unauthorized', () => {
  const auth = useAuthStore()
  auth.reset()
  router.push('/login')
})

export default router

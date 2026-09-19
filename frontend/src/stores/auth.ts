import { defineStore } from 'pinia'
import { api } from '../api/client'
import type { Identity, User } from '../types'

export const useAuthStore = defineStore('auth', {
  state: () => ({ identity: null as Identity | null, checked: false }),
  getters: {
    user: (state): User | undefined => state.identity?.user,
    isAdmin: (state) => state.identity?.role === 'admin',
    loggedIn: (state) => Boolean(state.identity),
  },
  actions: {
    async ensure() {
      if (this.checked) return this.loggedIn
      try {
        this.identity = await api<Identity>('/auth/me')
      } catch (_) {
        this.identity = null
      } finally {
        this.checked = true
      }
      return this.loggedIn
    },
    async login(username: string, password: string) {
      const result = await api<{ user: User }>('/auth/login', {
        method: 'POST', body: JSON.stringify({ username, password }),
      })
      this.identity = {
        subject: result.user.user_id,
        role: result.user.role,
        user: result.user,
        web_login_enabled: true,
      }
      this.checked = true
    },
    async logout() {
      try { await api('/auth/logout', { method: 'POST' }) } finally {
        this.identity = null
        this.checked = true
      }
    },
    reset() {
      this.identity = null
      this.checked = true
    },
  },
})

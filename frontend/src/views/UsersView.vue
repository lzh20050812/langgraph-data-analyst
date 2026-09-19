<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { Edit, Plus, Refresh } from '@element-plus/icons-vue'
import { api, errorMessage } from '../api/client'
import type { User } from '../types'
import { useAuthStore } from '../stores/auth'
import { useRouter } from 'vue-router'
const users=ref<User[]>([]), dialog=ref(false), loading=ref(false)
const auth=useAuthStore(), router=useRouter(), editing=ref<User|null>(null), saving=ref(false)
const form=reactive({username:'',display_name:'',password:'',role:'analyst' as 'analyst'|'admin',enabled:true})
const dialogTitle=computed(()=>editing.value?'编辑登录用户':'创建登录用户')
function reset(){editing.value=null;Object.assign(form,{username:'',display_name:'',password:'',role:'analyst',enabled:true})}
function openCreate(){reset();dialog.value=true}
function openEdit(user:User){editing.value=user;Object.assign(form,{username:user.username,display_name:user.display_name,password:'',role:user.role,enabled:user.enabled});dialog.value=true}
async function load(){loading.value=true;try{users.value=(await api<{users:User[]}>('/admin/users')).users}catch(error:unknown){ElMessage.error(errorMessage(error,'用户列表加载失败'))}finally{loading.value=false}}
async function save(){
  if(!form.display_name.trim()){ElMessage.warning('请输入显示名称');return}
  if(!editing.value&&!/^[\p{L}\p{N}_]{3,64}$/u.test(form.username)){ElMessage.warning('用户名需要 3–64 位字母、数字或下划线');return}
  if((!editing.value||form.password)&&(!/(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}/.test(form.password))){ElMessage.warning('密码至少 8 位，并包含大写字母、小写字母和数字');return}
  saving.value=true
  try{
    if(editing.value){
      const payload:Record<string,unknown>={display_name:form.display_name,role:form.role,enabled:form.enabled}
      if(form.password)payload.password=form.password
      const updated=await api<User>(`/admin/users/${editing.value.user_id}`,{method:'PATCH',body:JSON.stringify(payload)})
      const changedOwnPassword=updated.user_id===auth.user?.user_id&&Boolean(form.password)
      if(updated.user_id===auth.user?.user_id&&auth.identity)auth.identity.user=updated
      ElMessage.success('用户信息已更新')
      if(changedOwnPassword){dialog.value=false;auth.reset();await router.replace('/login');return}
    }else{
      await api('/admin/users',{method:'POST',body:JSON.stringify(form)})
      ElMessage.success('用户创建成功')
    }
    dialog.value=false;reset();await load()
  }catch(error:unknown){ElMessage.error(errorMessage(error,editing.value?'更新失败':'创建失败'))}finally{saving.value=false}
}
async function toggle(user:User,value:boolean){try{await api(`/admin/users/${user.user_id}`,{method:'PATCH',body:JSON.stringify({enabled:value})});ElMessage.success('状态已更新');load()}catch(error:unknown){ElMessage.error(errorMessage(error,'更新失败'));load()}}
onMounted(load)
</script>
<template>
  <div class="page">
    <div class="page-heading"><div><h2>用户管理</h2><p>管理浏览器登录账号、角色和启用状态</p></div><div><el-button :icon="Refresh" @click="load">刷新</el-button><el-button type="primary" :icon="Plus" @click="openCreate">新建用户</el-button></div></div>
    <div class="surface"><el-table v-loading="loading" :data="users" stripe>
      <el-table-column prop="display_name" label="用户"><template #default="s"><div class="user-cell"><el-avatar :size="34">{{s.row.display_name.slice(0,1)}}</el-avatar><div><strong>{{s.row.display_name}}</strong><small>@{{s.row.username}}</small></div></div></template></el-table-column>
      <el-table-column prop="role" label="角色" width="130"><template #default="s"><el-tag :type="s.row.role==='admin'?'danger':'info'">{{s.row.role==='admin'?'管理员':'分析员'}}</el-tag></template></el-table-column>
      <el-table-column label="最近登录" width="190"><template #default="s">{{s.row.last_login_at?new Date(s.row.last_login_at*1000).toLocaleString():'从未登录'}}</template></el-table-column>
      <el-table-column label="创建时间" width="190"><template #default="s">{{new Date(s.row.created_at*1000).toLocaleString()}}</template></el-table-column>
      <el-table-column label="启用" width="90"><template #default="s"><el-switch :model-value="s.row.enabled" :disabled="s.row.user_id===auth.user?.user_id" @change="(value: string | number | boolean) => toggle(s.row,Boolean(value))"/></template></el-table-column>
      <el-table-column label="操作" width="90" fixed="right"><template #default="s"><el-button link type="primary" :icon="Edit" @click="openEdit(s.row)">编辑</el-button></template></el-table-column>
    </el-table></div>
    <el-dialog v-model="dialog" :title="dialogTitle" width="470px" @closed="reset"><el-form label-position="top">
      <el-form-item label="用户名"><el-input v-model="form.username" :disabled="Boolean(editing)" placeholder="字母、数字或下划线"/></el-form-item>
      <el-form-item label="显示名称"><el-input v-model="form.display_name" maxlength="80" show-word-limit/></el-form-item>
      <el-form-item :label="editing?'重置密码（留空则不修改）':'初始密码'"><el-input v-model="form.password" type="password" show-password placeholder="至少 8 位"/></el-form-item>
      <el-form-item label="角色"><el-radio-group v-model="form.role" :disabled="editing?.user_id===auth.user?.user_id"><el-radio-button value="analyst">分析员</el-radio-button><el-radio-button value="admin">管理员</el-radio-button></el-radio-group></el-form-item>
      <el-form-item v-if="editing" label="账号状态"><el-switch v-model="form.enabled" :disabled="editing.user_id===auth.user?.user_id" active-text="启用" inactive-text="停用"/></el-form-item>
    </el-form><template #footer><el-button @click="dialog=false">取消</el-button><el-button type="primary" :loading="saving" @click="save">{{editing?'保存修改':'创建用户'}}</el-button></template></el-dialog>
  </div>
</template>
<style scoped>.user-cell{display:flex;align-items:center;gap:10px}.user-cell strong,.user-cell small{display:block}.user-cell small{color:#929bad;margin-top:3px}</style>

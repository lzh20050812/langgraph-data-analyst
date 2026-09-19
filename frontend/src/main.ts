import { createApp } from 'vue'
import { createPinia } from 'pinia'
import {
  ElAlert, ElAvatar, ElButton, ElDescriptions, ElDescriptionsItem, ElDialog,
  ElDrawer, ElDropdown, ElDropdownItem, ElDropdownMenu, ElForm, ElFormItem,
  ElIcon, ElInput, ElLoading, ElMenu, ElMenuItem, ElOption, ElRadioButton,
  ElRadioGroup, ElSelect, ElSkeleton, ElSwitch, ElTable, ElTableColumn, ElTag,
} from 'element-plus'
import 'element-plus/theme-chalk/base.css'
import 'element-plus/es/components/alert/style/css'
import 'element-plus/es/components/avatar/style/css'
import 'element-plus/es/components/button/style/css'
import 'element-plus/es/components/descriptions/style/css'
import 'element-plus/es/components/descriptions-item/style/css'
import 'element-plus/es/components/dialog/style/css'
import 'element-plus/es/components/drawer/style/css'
import 'element-plus/es/components/dropdown/style/css'
import 'element-plus/es/components/dropdown-item/style/css'
import 'element-plus/es/components/dropdown-menu/style/css'
import 'element-plus/es/components/form/style/css'
import 'element-plus/es/components/form-item/style/css'
import 'element-plus/es/components/icon/style/css'
import 'element-plus/es/components/input/style/css'
import 'element-plus/es/components/loading/style/css'
import 'element-plus/es/components/menu/style/css'
import 'element-plus/es/components/menu-item/style/css'
import 'element-plus/es/components/message/style/css'
import 'element-plus/es/components/message-box/style/css'
import 'element-plus/es/components/option/style/css'
import 'element-plus/es/components/radio-button/style/css'
import 'element-plus/es/components/radio-group/style/css'
import 'element-plus/es/components/select/style/css'
import 'element-plus/es/components/skeleton/style/css'
import 'element-plus/es/components/switch/style/css'
import 'element-plus/es/components/table/style/css'
import 'element-plus/es/components/table-column/style/css'
import 'element-plus/es/components/tag/style/css'
import App from './App.vue'
import router from './router'
import './styles.css'

const app = createApp(App)
const components = [
  ElAlert, ElAvatar, ElButton, ElDescriptions, ElDescriptionsItem, ElDialog,
  ElDrawer, ElDropdown, ElDropdownItem, ElDropdownMenu, ElForm, ElFormItem,
  ElIcon, ElInput, ElMenu, ElMenuItem, ElOption, ElRadioButton, ElRadioGroup,
  ElSelect, ElSkeleton, ElSwitch, ElTable, ElTableColumn, ElTag,
]
for (const component of components) app.component(component.name!, component)
app.use(ElLoading).use(createPinia()).use(router).mount('#app')

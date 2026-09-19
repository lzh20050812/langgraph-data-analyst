export interface User {
  user_id: string
  username: string
  display_name: string
  role: 'analyst' | 'admin'
  enabled: boolean
  created_at: number
  last_login_at?: number | null
}

export interface Identity {
  subject: string
  role: 'analyst' | 'admin'
  user?: User
  web_login_enabled: boolean
}

export interface Conversation {
  conversation_id: string
  owner_id: string
  title: string
  created_at: number
  updated_at: number
  context_revision?: number
  analysis_context?: AnalysisRequest | null
  pending_clarification?: AnalysisRequest | null
  pending_task_id?: string | null
}

export interface AnalysisRequest {
  schema_version: string
  metric_catalog_version: string
  metrics: string[]
  dimensions: string[]
  time_scope: { field?: string | null; years: number[]; start?: string | null; end?: string | null }
  filters: Record<string, string[]>
  sort: { field: string; direction: 'asc' | 'desc' }[]
  limit?: number | null
  data_source: string
  unresolved_ambiguities: string[]
  unsupported_conditions: string[]
}

export interface Message {
  message_id: number
  conversation_id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  task_id?: string | null
  created_at: number
  metadata?: Record<string, any> | null
}

export interface ChartDefinition {
  id?: string
  title?: string
  type: 'bar' | 'line' | 'pie' | 'scatter' | 'dashboard' | string
  option: Record<string, any>
}

export interface EvidenceCheck {
  code: string
  status: 'verified' | 'failed' | 'unverifiable' | string
  message: string
  details?: Record<string, any>
}

export interface EvidenceBundle {
  sql?: string | null
  metric_catalog_version?: string | null
  source_tables?: string[]
  facts?: Array<{ fact_id: string; label: string; value: any; unit?: string | null; dimensions?: Record<string, any> }>
  validation?: { status?: string; checks?: EvidenceCheck[] }
  report_validation?: { status?: string; checks?: EvidenceCheck[] }
  limitations?: string[]
  provenance?: Record<string, any>
}

export interface TaskResult {
  request_id?: string
  success?: boolean
  sql?: string | null
  report?: string | null
  charts?: ChartDefinition[] | null
  query_result?: Record<string, any>[]
  query_result_total_rows?: number
  query_result_truncated?: boolean
  evidence?: EvidenceBundle | null
  [key: string]: any
}

export interface Task {
  task_id: string
  query: string
  intent?: string
  status: string
  error?: string | null
  created_at: number
  updated_at: number
  result?: TaskResult | null
  parent_task_id?: string | null
  retry_mode?: string | null
  retry_risks?: string[]
  trace_id?: string | null
}

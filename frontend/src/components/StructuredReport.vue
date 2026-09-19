<script setup lang="ts">
import { computed } from 'vue'

type Block = { type: 'heading' | 'paragraph' | 'unordered' | 'ordered'; text?: string; items?: string[]; level?: number }
const props = defineProps<{ content: string }>()

const blocks = computed<Block[]>(() => {
  const lines = (props.content || '').replace(/\r/g, '').split('\n')
  const result: Block[] = []
  let paragraph: string[] = []
  let list: Block | undefined
  const flushParagraph = () => {
    const text = paragraph.join(' ').trim()
    if (text) result.push({ type: 'paragraph', text })
    paragraph = []
  }
  const flushList = () => { if (list) result.push(list); list = undefined }
  for (const raw of lines) {
    const line = raw.trim()
    if (!line) { flushParagraph(); flushList(); continue }
    const heading = /^(#{1,4})\s+(.+)$/.exec(line)
    const unordered = /^[-*•]\s+(.+)$/.exec(line)
    const ordered = /^\d+[.)]\s+(.+)$/.exec(line)
    if (heading) {
      flushParagraph(); flushList()
      result.push({ type: 'heading', level: heading[1].length, text: heading[2] })
    } else if (unordered || ordered) {
      flushParagraph()
      const type = unordered ? 'unordered' : 'ordered'
      if (!list || list.type !== type) { flushList(); list = { type, items: [] } }
      list.items!.push((unordered || ordered)![1])
    } else {
      flushList(); paragraph.push(line)
    }
  }
  flushParagraph(); flushList()
  return result
})
</script>

<template>
  <div class="structured-report">
    <template v-for="(block,index) in blocks" :key="index">
      <h3 v-if="block.type==='heading'" :class="`level-${block.level}`">{{ block.text }}</h3>
      <p v-else-if="block.type==='paragraph'">{{ block.text }}</p>
      <ul v-else-if="block.type==='unordered'"><li v-for="item in block.items" :key="item">{{ item }}</li></ul>
      <ol v-else><li v-for="item in block.items" :key="item">{{ item }}</li></ol>
    </template>
  </div>
</template>

<style scoped>
.structured-report{font-size:14px;line-height:1.75;color:#344056}.structured-report h3{margin:15px 0 5px;color:#202c42;font-size:14px}.structured-report h3:first-child{margin-top:0}.structured-report .level-1{font-size:16px;border-left:3px solid #4f6ef7;padding-left:8px}.structured-report p{margin:5px 0}.structured-report ul,.structured-report ol{margin:5px 0;padding-left:22px}.structured-report li+li{margin-top:3px}
</style>

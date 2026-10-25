<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, watch, ref } from 'vue'
import { useData, useRoute, withBase } from 'vitepress'
import RelatedArticles from './components/RelatedArticles.vue'

const { page, frontmatter, isDark } = useData()
const route = useRoute()
const isArticle = computed(() => frontmatter.value.layout === 'article')
const isAbout = computed(() => frontmatter.value.layout === 'about')
const headings = ref<{ title: string; id: string; depth: number }[]>([])
const mobileOpen = ref(false)
const activeHeading = ref('')
let headingElements: HTMLElement[] = []
let scrollFrame = 0
let bodyObserver: ResizeObserver | undefined

function updateActiveHeading() {
  scrollFrame = 0
  if (!headingElements.length) { activeHeading.value = ''; return }
  let current = headingElements[0]
  for (const element of headingElements) {
    if (element.getBoundingClientRect().top <= 110) current = element
    else break
  }
  if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 8) {
    current = headingElements[headingElements.length - 1]
  }
  activeHeading.value = current.id
}

function scheduleHeadingUpdate() {
  if (!scrollFrame) scrollFrame = window.requestAnimationFrame(updateActiveHeading)
}

async function updateHeadings() {
  await nextTick()
  bodyObserver?.disconnect()
  const elements = [...document.querySelectorAll<HTMLElement>('.article-body h1[id], .article-body h2[id], .article-body h3[id]')].slice(1)
  headingElements = elements
  const baseLevel = Math.min(...elements.map(element => Number(element.tagName.slice(1))))
  headings.value = elements.map(element => ({
      id: element.id,
      title: (element.textContent ?? '').replace(/\u200b/g, '').trim(),
      depth: Number(element.tagName.slice(1)) - baseLevel,
    }))
  const body = document.querySelector('.article-body')
  if (body) bodyObserver?.observe(body)
  scheduleHeadingUpdate()
}
onMounted(() => {
  bodyObserver = new ResizeObserver(scheduleHeadingUpdate)
  updateHeadings()
  window.addEventListener('scroll', scheduleHeadingUpdate, { passive: true })
  window.addEventListener('resize', scheduleHeadingUpdate)
  watch(isDark, dark => {
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', dark ? '#1b1e1b' : '#f8f7f3')
  }, { immediate: true })
})
onUnmounted(() => {
  bodyObserver?.disconnect()
  window.cancelAnimationFrame(scrollFrame)
  window.removeEventListener('scroll', scheduleHeadingUpdate)
  window.removeEventListener('resize', scheduleHeadingUpdate)
})
watch(() => page.value.relativePath, () => { mobileOpen.value = false; updateHeadings() }, { flush: 'post' })
watch(activeHeading, async () => {
  await nextTick()
  const outline = document.querySelector<HTMLElement>('.article-outline')
  const active = outline?.querySelector<HTMLElement>('[aria-current="location"]')
  if (!outline || !active) return
  const parent = outline.getBoundingClientRect()
  const child = active.getBoundingClientRect()
  const behavior = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'
  if (child.bottom > parent.bottom) outline.scrollBy({ top: child.bottom - parent.bottom + 12, behavior })
  else if (child.top < parent.top) outline.scrollBy({ top: child.top - parent.top - 12, behavior })
})
</script>

<template>
  <div class="workshop-shell" :class="{ 'is-home': route.path === withBase('/'), 'is-listing': route.path === withBase('/articles.html') }">
    <header class="site-header">
      <a class="brand" :href="withBase('/')" aria-label="Moonnryan's Research Workshop 首页">
        <span class="brand-mark" aria-hidden="true"></span>
        <span class="brand-name">moonnryan<span class="brand-subtitle">research workshop</span></span>
      </a>
      <button class="menu-toggle" :aria-expanded="mobileOpen" aria-controls="main-navigation" @click="mobileOpen = !mobileOpen">菜单</button>
      <nav id="main-navigation" class="main-nav" :class="{ 'is-open': mobileOpen }" aria-label="主导航">
        <a :class="{ active: route.path === withBase('/') }" :href="withBase('/')">首页</a>
        <a :class="{ active: frontmatter.layout === 'topics' }" :href="withBase('/topics.html')">主题</a>
        <a :class="{ active: isArticle || route.path.includes('/articles') }" :href="withBase('/articles.html')">文章</a>
        <a :class="{ active: isAbout }" :href="withBase('/about.html')">关于</a>
        <a class="github-link" href="https://github.com/moonnryan" target="_blank" rel="noreferrer">GitHub <span aria-hidden="true">↗</span></a>
        <button class="appearance-toggle" type="button" aria-label="切换深浅色模式" title="切换深浅色模式" @click="isDark = !isDark">
          <svg class="theme-moon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20.5 14A8.5 8.5 0 0 1 10 3.5 8.5 8.5 0 1 0 20.5 14Z" /></svg>
          <svg class="theme-sun" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4" /><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5" /></svg>
        </button>
      </nav>
    </header>

    <main id="main-content">
      <div v-if="isArticle" class="article-layout">
        <article class="article-main">
          <a class="back-link" :href="withBase('/articles.html')">← 所有文章</a>
          <div class="article-meta">
            <span>{{ frontmatter.categoryName }}</span><span class="meta-dot">·</span>
            <span>约 {{ frontmatter.minutes }} 分钟</span>
          </div>
          <div v-if="frontmatter.tags?.length" class="article-tags article-page-tags" aria-label="文章标签">
            <a v-for="tag in frontmatter.tags" :key="tag" :href="withBase('/articles.html?tag=' + encodeURIComponent(tag))">{{ tag }}</a>
          </div>
          <Content :key="page.relativePath" class="vp-doc article-body" />
        </article>
        <aside v-if="headings.length" class="article-outline" aria-label="文章目录">
          <div class="outline-label">本文目录</div>
          <a v-for="heading in headings" :key="heading.id" :href="'#' + heading.id" :class="['outline-depth-' + heading.depth, { 'is-current': heading.id === activeHeading }]" :aria-current="heading.id === activeHeading ? 'location' : undefined">{{ heading.title }}</a>
        </aside>
        <RelatedArticles :url="route.path" />
      </div>
      <div v-else-if="isAbout" class="about-layout">
        <aside class="about-profile" aria-label="个人资料">
          <img v-if="frontmatter.profile?.avatar" class="about-avatar" :src="withBase(frontmatter.profile.avatar)" :alt="frontmatter.profile.name || 'moonnryan'">
          <div v-else class="about-monogram" aria-hidden="true"><span class="brand-mark"></span></div>
          <p class="about-profile-name">{{ frontmatter.profile?.name || 'moonnryan' }}</p>
          <p v-if="frontmatter.profile?.tagline" class="about-profile-tagline">{{ frontmatter.profile.tagline }}</p>
          <nav v-if="frontmatter.profile?.links?.length" class="about-profile-links" aria-label="个人链接">
            <a v-for="link in frontmatter.profile.links" :key="link.url" :href="link.url" :target="link.url.startsWith('https://') ? '_blank' : undefined" :rel="link.url.startsWith('https://') ? 'noreferrer' : undefined" :aria-label="link.label" :title="link.label">
              <svg v-if="link.icon === 'github'" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .75a11.25 11.25 0 0 0-3.558 21.923c.563.104.768-.244.768-.542 0-.267-.01-.975-.015-1.913-3.13.68-3.791-1.508-3.791-1.508-.512-1.3-1.25-1.646-1.25-1.646-1.023-.7.078-.686.078-.686 1.132.08 1.727 1.162 1.727 1.162 1.006 1.724 2.64 1.226 3.283.938.102-.729.394-1.226.715-1.508-2.499-.284-5.126-1.25-5.126-5.56 0-1.228.44-2.232 1.16-3.019-.117-.285-.503-1.43.11-2.98 0 0 .945-.303 3.095 1.153A10.79 10.79 0 0 1 12 6.185c.956.005 1.918.13 2.817.379 2.148-1.456 3.091-1.153 3.091-1.153.614 1.55.228 2.695.112 2.98.722.787 1.158 1.791 1.158 3.019 0 4.322-2.631 5.273-5.138 5.552.404.35.765 1.04.765 2.096 0 1.513-.014 2.733-.014 3.104 0 .3.203.65.774.54A11.252 11.252 0 0 0 12 .75Z"/></svg>
              <svg v-else-if="link.icon === 'orcid'" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="12" fill="#a6ce39"/><g fill="#fff"><circle cx="6.7" cy="6.5" r="1"/><path d="M5.9 9h1.6v9H5.9zM10 5.7h4.1c4.1 0 6.2 2.6 6.2 6.1 0 3.7-2.3 6.2-6.3 6.2h-4zm1.7 1.6v9.1H14c3 0 4.6-1.8 4.6-4.6 0-2.7-1.6-4.5-4.6-4.5z"/></g></svg>
              <svg v-else-if="link.icon === 'email'" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><rect x="2.5" y="4.5" width="19" height="15" rx="2"/><path d="m3 6 9 7 9-7"/></svg>
              <span v-else>{{ link.label }} <span aria-hidden="true">↗</span></span>
            </a>
          </nav>
          <a v-if="frontmatter.profile?.email" class="about-profile-email" :href="'mailto:' + frontmatter.profile.email">{{ frontmatter.profile.email }}</a>
        </aside>
        <div class="about-content"><Content class="vp-doc" /></div>
      </div>
      <Content v-else :key="page.relativePath" />
    </main>

    <footer class="site-footer">
      <span>Copyright © 2026. Mengran.Yang.</span>
      <a class="footer-note" href="https://vitepress.dev/" target="_blank" rel="noreferrer">Powered by VitePress ↗</a>
    </footer>
  </div>
</template>

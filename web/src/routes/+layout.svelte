<script lang="ts">
  import '../app.css';
  import { onMount, type Snippet } from 'svelte';
  import { page } from '$app/state';
  import { resolve } from '$app/paths';
  import { streams } from '$lib/stores/streams.svelte';
  import { health } from '$lib/stores/health.svelte';
  import { config } from '$lib/stores/config.svelte';
  import HealthBadge from '$lib/components/HealthBadge.svelte';

  interface Props {
    children?: Snippet;
  }
  let { children }: Props = $props();

  const TABS = [
    { href: resolve('/'), label: 'Dashboard' },
    { href: resolve('/workspace'), label: 'Workspace' },
    { href: resolve('/converter'), label: 'Converter' }
  ];

  function isActive(href: string): boolean {
    const root = resolve('/');
    if (href === root) return page.url.pathname === root;
    return page.url.pathname === href || page.url.pathname.startsWith(href + '/');
  }

  onMount(() => {
    streams.start();
    health.start();
    void config.refresh();
    return () => {
      streams.stop();
      health.stop();
    };
  });
</script>

<div class="flex min-h-screen flex-col">
  <header class="border-b border-zinc-200 bg-white">
    <div class="mx-auto flex h-14 max-w-7xl items-center justify-between gap-4 px-4">
      <div class="flex items-center gap-6">
        <a href={resolve('/')} class="flex items-center gap-2 text-zinc-900">
          <span class="inline-block h-2.5 w-2.5 rounded-full bg-blue-500"></span>
          <span class="text-base font-semibold tracking-tight">AcousticsLab</span>
        </a>
        <nav class="flex items-center gap-1">
          {#each TABS as tab (tab.href)}
            <a
              href={tab.href}
              class="rounded-md px-3 py-1.5 text-sm font-medium transition"
              class:bg-zinc-100={isActive(tab.href)}
              class:text-zinc-900={isActive(tab.href)}
              class:text-zinc-500={!isActive(tab.href)}
              class:hover:text-zinc-900={!isActive(tab.href)}>{tab.label}</a
            >
          {/each}
        </nav>
      </div>

      <div class="flex items-center gap-3">
        <HealthBadge />
      </div>
    </div>
  </header>

  <main class="mx-auto w-full max-w-7xl flex-1 px-4 py-6">
    {@render children?.()}
  </main>
</div>

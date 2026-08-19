# Cicerone website (Astro Starlight)

Docs site for [cicerone.dev](https://cicerone.dev). Built with
[Starlight](https://starlight.astro.build/); markdown under repo `docs/` is
synced into the Starlight content collection at build time. Articles are the
same static build (no CMS): posts live under `src/content/docs/articles/`.
With no published post, the articles plugin is off — no header link, RSS, or
`/articles/` route.

## Commands

```sh
cd website
npm ci
npm run dev      # sync docs/ + local preview
npm run build    # sync docs/ + production build → dist/
npm run preview  # serve dist/
```

## Layout

| Path | Role |
| --- | --- |
| `src/content/docs/index.mdx` | Landing (Starlight splash) |
| `src/content/docs/articles/` | Articles (`title` + `date` frontmatter) → `/articles/` |
| `scripts/sync-docs.mjs` | Copies `../docs/*.md` → `src/content/docs/` with frontmatter |
| `astro.config.mjs` | Site URL, sidebar, logo, social, articles plugin |
| `public/CNAME` | Custom domain (`cicerone.dev`) |
| `public/images/` | Site diagrams (`flow.svg`) |
| `public/images/docs/` | Copied from `../docs/images/` at build time (gitignored) |

Generated `src/content/docs/how-it-works.md`, `tutorial.md`,
`architecture.md`, `incremental-events.md`, and `public/images/docs/` are
gitignored; they are created at build/dev time. CI and local builds always
sync from `docs/`. Articles are **not** synced from `docs/` — add Markdown
under `src/content/docs/articles/` (see below).

## Articles

Jekyll-style: a `.md` file with YAML frontmatter, HTML at build time, nothing
dynamic. Author globally is `nicholas` (`astro.config.mjs`). Until a
non-draft post exists, Articles is omitted from the build.

```md
---
title: Post title
description: One-line summary for search results and Open Graph.
date: 2026-08-19
excerpt: Listing blurb (falls back to description / body).
authors:
  - nicholas
---

Body…
```

Drafts (`draft: true`, or YAML 1.1 `yes` / `on`) are omitted from
production builds; `astro dev` still loads them so `/articles/` can be
previewed. Frontmatter is parsed as YAML. RSS is
`/articles/rss.xml` once a post is published. Article pages use IBM Plex
Serif at ~65ch; chrome stays IBM Plex Sans, dates IBM Plex Mono.

## Publishing

[`.github/workflows/pages.yml`](../.github/workflows/pages.yml) builds
`website/` on PRs and on pushes to `main` that touch `website/**` or
`docs/**`. Only `main` deploys. Website-only PRs skip Docker lint/pytest;
the `ci` job still succeeds so a required check is not left pending.

**One-time:** Settings → Pages → Source = **GitHub Actions**, custom domain
`cicerone.dev`. DNS notes for Gandi apex records are below.

### DNS (apex `cicerone.dev`)

| Type | Name | Value |
| --- | --- | --- |
| `A` | `@` | `185.199.108.153` |
| `A` | `@` | `185.199.109.153` |
| `A` | `@` | `185.199.110.153` |
| `A` | `@` | `185.199.111.153` |
| `AAAA` | `@` | `2606:50c0:8000::153` |
| `AAAA` | `@` | `2606:50c0:8001::153` |
| `AAAA` | `@` | `2606:50c0:8002::153` |
| `AAAA` | `@` | `2606:50c0:8003::153` |
| `CNAME` | `www` | `torbido-hq.github.io` |

Remove Gandi web-forward / `webredir` records first.

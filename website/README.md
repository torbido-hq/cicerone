# Cicerone website (Astro Starlight)

Docs site for [cicerone.dev](https://cicerone.dev). Built with
[Starlight](https://starlight.astro.build/); markdown under repo `docs/` is
synced into the Starlight content collection at build time.

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
| `scripts/sync-docs.mjs` | Copies `../docs/*.md` → `src/content/docs/` with frontmatter |
| `astro.config.mjs` | Site URL, sidebar, logo, social |
| `public/CNAME` | Custom domain (`cicerone.dev`) |
| `public/images/` | Screenshots / diagrams |

Generated `src/content/docs/tutorial.md` and `architecture.md` are gitignored;
CI and local builds always sync from `docs/`.

## Publishing

[`.github/workflows/pages.yml`](../.github/workflows/pages.yml) builds
`website/` on pushes to `main` that touch `website/**` or `docs/**`.

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

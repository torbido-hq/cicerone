# Cicerone website

Static project site for [cicerone.dev](https://cicerone.dev) via
[GitHub Pages](https://docs.github.com/en/pages). Source lives here; CI
compiles Tailwind and renders `docs/*.md` into `dist/`, then deploys that
folder (including `CNAME`).

## Layout

| Path | Role |
| --- | --- |
| `index.html` | Landing (brand + dashboard screenshot hero) |
| `scripts/build-docs.mjs` | Renders `../docs/*.md` → `dist/docs/*.html` |
| `templates/layout.html` | Shared chrome for rendered docs |
| `css/input.css` | Tailwind v4 theme + typography plugin |
| `robots.txt` / `sitemap.xml` | SEO crawl hints (sitemap written at build) |
| `CNAME` | Custom domain (`cicerone.dev`) |
| `images/` | Screenshots and diagrams |
| `assets/` | Logo / favicon |
| `dist/` | Build output (gitignored) |

Markdown under repo `docs/` is the source of truth. The site rebuilds
Tutorial and Architecture HTML from those files on every Pages deploy.

## Preview locally

Needs Node 22+:

```sh
cd website
npm ci
npm run build
python -m http.server 4173 --directory dist
```

Then visit `http://127.0.0.1:4173/` (docs at `/docs/`).

## Publishing

[`.github/workflows/pages.yml`](../.github/workflows/pages.yml) runs
`npm ci && npm run build` in `website/` on pushes to `main` that touch
`website/**` or `docs/**` (and on `workflow_dispatch`), then uploads
`website/dist`.

### One-time GitHub settings

1. **Settings → Pages → Build and deployment → Source = GitHub Actions**
2. **Custom domain** = `cicerone.dev`, then **Save** (check **Enforce HTTPS**
   after DNS verifies)

For Actions-based Pages, the domain is owned by that Settings field (a
`CNAME` in the artifact is not required). We still ship `website/CNAME` so
the intended hostname is obvious in-repo and survives a switch to
branch-based publishing.

### DNS (apex `cicerone.dev`)

At the registrar / DNS host, point the apex at GitHub Pages. Either:

**A / AAAA** (GitHub’s current Pages IPs — confirm in
[GitHub’s docs](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/managing-a-custom-domain-for-your-github-pages-site)):

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

Optional www:

| Type | Name | Value |
| --- | --- | --- |
| `CNAME` | `www` | `torbido-hq.github.io` |

If the DNS host supports **ALIAS / ANAME** for apex, that can target
`torbido-hq.github.io` instead of the A records.

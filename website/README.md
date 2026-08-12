# Cicerone website

Static project site for [cicerone.dev](https://cicerone.dev) via
[GitHub Pages](https://docs.github.com/en/pages). Source lives here; CI
compiles Tailwind into `dist/` and deploys that folder (including `CNAME`).

## Layout

| Path | Role |
| --- | --- |
| `index.html` | Landing (brand + dashboard screenshot hero) |
| `documentation.html` | Getting started, serve API, dashboard |
| `architecture.html` | Package / pipeline overview |
| `css/input.css` | Tailwind v4 theme + typography plugin |
| `CNAME` | Custom domain (`cicerone.dev`) |
| `images/` | Screenshots and diagrams |
| `assets/` | Logo / favicon |
| `dist/` | Build output (gitignored) |

Markdown under `docs/` remains the detailed source of truth; pages here
summarize and link back to the repo.

## Preview locally

Needs Node 22+:

```sh
cd website
npm ci
npm run build
python -m http.server 4173 --directory dist
```

Then visit `http://127.0.0.1:4173/`. For CSS iteration, run `npm run build`
again (or `npm run build:css` after an initial `npm run build`).

## Publishing

[`.github/workflows/pages.yml`](../.github/workflows/pages.yml) runs
`npm ci && npm run build` in `website/` on pushes to `main` that touch
`website/**` (and on `workflow_dispatch`), then uploads `website/dist`.

### One-time GitHub settings

1. **Settings → Pages → Build and deployment → Source = GitHub Actions**
2. **Custom domain** = `cicerone.dev` (check **Enforce HTTPS** after DNS
   verifies)

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

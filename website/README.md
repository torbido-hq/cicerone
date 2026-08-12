# Cicerone website

Static project site for [GitHub Pages](https://docs.github.com/en/pages)
(`https://torbido-hq.github.io/cicerone/`). Source lives here; CI compiles
Tailwind into `dist/` and deploys that folder.

## Layout

| Path | Role |
| --- | --- |
| `index.html` | Landing (brand + dashboard screenshot hero) |
| `documentation.html` | Getting started, serve API, dashboard |
| `architecture.html` | Package / pipeline overview |
| `css/input.css` | Tailwind v4 theme + typography plugin |
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

**One-time repo setting:** Settings → Pages → Build and deployment → Source
= **GitHub Actions**. Until that is set, the workflow uploads an artifact but
Pages will not serve the site.

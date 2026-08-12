# Cicerone website

Static project site for [GitHub Pages](https://docs.github.com/en/pages)
(`https://torbido-hq.github.io/cicerone/`).

## Layout

| Path | Role |
| --- | --- |
| `index.html` | Landing (brand + dashboard screenshot hero) |
| `documentation.html` | Getting started, serve API, dashboard |
| `architecture.html` | Package / pipeline overview |
| `images/` | Screenshots and diagrams |
| `assets/` | Logo / favicon (copied from `src/cicerone/static/`) |

Markdown under `docs/` remains the detailed source of truth; pages here
summarize and link back to the repo.

## Preview locally

Open `index.html` in a browser, or from the repo root:

```sh
python -m http.server 4173 --directory website
```

Then visit `http://127.0.0.1:4173/`.

## Publishing

[`.github/workflows/pages.yml`](../.github/workflows/pages.yml) deploys this
folder on pushes to `main` that touch `website/**` (and on
`workflow_dispatch`).

**One-time repo setting:** Settings → Pages → Build and deployment → Source
= **GitHub Actions**. Until that is set, the workflow uploads an artifact but
Pages will not serve the site.

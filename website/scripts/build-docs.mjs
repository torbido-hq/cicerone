#!/usr/bin/env node
/**
 * Render repo docs/*.md into dist/docs/*.html using the site layout.
 * Run from website/ via `npm run build:docs` (after dist/ exists or creates docs/).
 */
import {
  copyFileSync,
  cpSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import MarkdownIt from "markdown-it";
import markdownItAnchor from "markdown-it-anchor";

const __dirname = dirname(fileURLToPath(import.meta.url));
const websiteRoot = resolve(__dirname, "..");
const repoRoot = resolve(websiteRoot, "..");
const docsSrc = join(repoRoot, "docs");
const distRoot = join(websiteRoot, "dist");
const distDocs = join(distRoot, "docs");
const layoutPath = join(websiteRoot, "templates", "layout.html");
const githubBlob = "https://github.com/torbido-hq/cicerone/blob/main";

const DOC_PAGES = [
  {
    source: "tutorial.md",
    out: "tutorial.html",
    title: "Tutorial",
    description: "Hands-on walkthrough from sample data to serve API and dashboard.",
  },
  {
    source: "architecture.md",
    out: "architecture.html",
    title: "Architecture",
    description: "How Cicerone packages fit together: I/O, model, job, serve, dashboard.",
  },
];

function githubSlugify(s) {
  return String(s)
    .trim()
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-");
}

const md = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: false,
}).use(markdownItAnchor, {
  level: [1, 2, 3, 4],
  slugify: githubSlugify,
  permalink: markdownItAnchor.permalink.linkInsideHeader({
    symbol: "#",
    placement: "before",
    class: "header-anchor not-prose mr-2 text-cyan no-underline opacity-0 transition group-hover:opacity-100 focus:opacity-100",
    ariaHidden: false,
  }),
  // markdown-it-anchor wraps headings; add group for hover permalink
  callback(token) {
    token.attrJoin("class", "group scroll-mt-24");
  },
});

function rewriteMarkdown(source, fileName) {
  let text = source;

  // Drop the leading logo — site chrome already brands every page.
  text = text.replace(
    /^\s*<img\s+src="\.\.\/src\/cicerone\/static\/cicerone-logo\.svg"[^>]*>\s*\n+/i,
    "",
  );

  // Repo-root markdown links → GitHub blob.
  text = text.replace(/\]\(\.\.\/([^)#]+)(#[^)]*)?\)/g, (_m, path, hash = "") => {
    return `](${githubBlob}/${path}${hash})`;
  });

  // Sibling .md links → rendered HTML in this folder.
  text = text.replace(/\]\(([^)/]+\.md)(#[^)]*)?\)/g, (_m, name, hash = "") => {
    const html = name.replace(/\.md$/, ".html");
    return `](${html}${hash})`;
  });

  // Any remaining in-repo static asset refs from markdown HTML.
  text = text.replace(
    /\.\.\/src\/cicerone\/static\//g,
    "../assets/",
  );

  // Image paths that point at docs/images from within docs/.
  text = text.replace(/\]\(images\//g, "](../images/");
  text = text.replace(/src="images\//g, 'src="../images/');

  void fileName;
  return text;
}

function renderLayout({ title, description, content, root, docsCurrent }) {
  let html = readFileSync(layoutPath, "utf8");
  html = html
    .replaceAll("{{TITLE}}", title)
    .replaceAll("{{DESCRIPTION}}", description)
    .replaceAll("{{ROOT}}", root)
    .replaceAll("{{CONTENT}}", content)
    .replaceAll(
      "{{DOCS_NAV_CLASS}}",
      docsCurrent ? "text-deep" : "",
    )
    .replaceAll(
      "{{DOCS_CURRENT}}",
      docsCurrent ? 'aria-current="page"' : "",
    );
  return html;
}

function pageShell(inner) {
  return `<div class="py-12 sm:py-14">${inner}</div>`;
}

function buildDocPage(page) {
  const srcPath = join(docsSrc, page.source);
  const raw = readFileSync(srcPath, "utf8");
  const body = md.render(rewriteMarkdown(raw, page.source));
  const content = pageShell(`
    <p class="mb-3 font-display text-eyebrow uppercase text-cyan">Docs · from <code class="rounded bg-deep/10 px-1.5 py-0.5 font-mono text-[0.85em] normal-case tracking-normal">docs/${page.source}</code></p>
    <article class="prose-cicerone">
      ${body}
    </article>
    <p class="mt-12 text-base text-muted">
      Source:
      <a href="${githubBlob}/docs/${page.source}">docs/${page.source}</a>
      on GitHub.
    </p>
  `);
  const html = renderLayout({
    title: page.title,
    description: page.description,
    content,
    root: "../",
    docsCurrent: true,
  });
  writeFileSync(join(distDocs, page.out), html);
  return page;
}

function buildDocsIndex(pages) {
  const cards = pages
    .map(
      (p) => `
      <li class="border-b border-line py-6">
        <a class="font-display text-2xl font-bold tracking-tight text-ink no-underline hover:text-cyan" href="${p.out}">${p.title}</a>
        <p class="mt-2 max-w-[44ch] text-lede text-muted">${p.description}</p>
        <p class="mt-2 font-mono text-sm text-muted">docs/${p.source}</p>
      </li>`,
    )
    .join("\n");

  const content = pageShell(`
    <h1 class="mb-3 font-display text-page text-ink">Documentation</h1>
    <p class="mb-8 max-w-[44ch] text-lede text-muted">Rendered from the markdown under <code class="rounded bg-deep/10 px-1.5 py-0.5 font-mono text-[0.9em]">docs/</code> in the repository. Edit those files — this site rebuilds them on deploy.</p>
    <ul class="list-none p-0">${cards}</ul>
    <div class="mt-10 border-l-[3px] border-cyan bg-cyan/10 px-4 py-3 text-ink-soft">
      Serve OpenAPI schema:
      <a href="openapi/serve.openapi.json">docs/openapi/serve.openapi.json</a>
    </div>
  `);

  writeFileSync(
    join(distDocs, "index.html"),
    renderLayout({
      title: "Documentation",
      description: "Cicerone documentation rendered from docs/ markdown.",
      content,
      root: "../",
      docsCurrent: true,
    }),
  );
}

function copyDocsAssets() {
  mkdirSync(join(distRoot, "images"), { recursive: true });
  const imgDir = join(docsSrc, "images");
  if (existsSync(imgDir)) {
    for (const name of readdirSync(imgDir)) {
      copyFileSync(join(imgDir, name), join(distRoot, "images", name));
    }
  }
  const openapiSrc = join(docsSrc, "openapi");
  if (existsSync(openapiSrc)) {
    cpSync(openapiSrc, join(distDocs, "openapi"), { recursive: true });
  }
}

function main() {
  mkdirSync(distDocs, { recursive: true });
  copyDocsAssets();
  const built = DOC_PAGES.map(buildDocPage);
  buildDocsIndex(built);
  console.log(`Rendered ${built.length} docs pages → dist/docs/`);
}

main();

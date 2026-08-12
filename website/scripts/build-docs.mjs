#!/usr/bin/env node
/**
 * Render repo docs/*.md into dist/docs/*.html and write sitemap.xml.
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
const siteOrigin = "https://cicerone.dev";

const DOC_PAGES = [
  {
    source: "tutorial.md",
    out: "tutorial.html",
    path: "/docs/tutorial.html",
    nav: "tutorial",
    title: "Tutorial",
    description:
      "Hands-on Cicerone walkthrough: sample data, batch job, serve API, and dashboard.",
  },
  {
    source: "architecture.md",
    out: "architecture.html",
    path: "/docs/architecture.html",
    nav: "architecture",
    title: "Architecture",
    description:
      "How Cicerone packages fit together: I/O, model strategies, job loop, serve, and dashboard.",
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
    class: "header-anchor",
    ariaHidden: false,
    renderAttrs: () => ({
      "aria-label": "Permalink to this section",
    }),
  }),
  callback(token) {
    token.attrJoin("class", "group scroll-mt-20");
  },
});

function rewriteMarkdown(source) {
  let text = source;

  text = text.replace(
    /^\s*<img\s+src="\.\.\/src\/cicerone\/static\/cicerone-logo\.svg"[^>]*>\s*\n+/i,
    "",
  );

  text = text.replace(/\]\(\.\.\/([^)#]+)(#[^)]*)?\)/g, (_m, path, hash = "") => {
    return `](${githubBlob}/${path}${hash})`;
  });

  text = text.replace(/\]\(([^)/]+\.md)(#[^)]*)?\)/g, (_m, name, hash = "") => {
    const html = name.replace(/\.md$/, ".html");
    return `](${html}${hash})`;
  });

  text = text.replace(/\.\.\/src\/cicerone\/static\//g, "../assets/");
  text = text.replace(/\]\(images\//g, "](../images/");
  text = text.replace(/src="images\//g, 'src="../images/');

  return text;
}

function currentAttr(active, key) {
  return active === key ? 'aria-current="page"' : "";
}

function jsonLd(obj) {
  return JSON.stringify(obj).replace(/</g, "\\u003c");
}

function renderLayout({
  title,
  description,
  content,
  root,
  canonical,
  ogType,
  nav,
  structuredData,
}) {
  let html = readFileSync(layoutPath, "utf8");
  return html
    .replaceAll("{{TITLE}}", title)
    .replaceAll("{{DESCRIPTION}}", description)
    .replaceAll("{{ROOT}}", root)
    .replaceAll("{{CONTENT}}", content)
    .replaceAll("{{CANONICAL}}", canonical)
    .replaceAll("{{OG_TYPE}}", ogType)
    .replaceAll("{{JSON_LD}}", jsonLd(structuredData))
    .replaceAll("{{HOME_CURRENT}}", currentAttr(nav, "home"))
    .replaceAll("{{DOCS_CURRENT}}", currentAttr(nav, "docs"))
    .replaceAll("{{TUTORIAL_CURRENT}}", currentAttr(nav, "tutorial"))
    .replaceAll("{{ARCHITECTURE_CURRENT}}", currentAttr(nav, "architecture"));
}

function buildDocPage(page) {
  const raw = readFileSync(join(docsSrc, page.source), "utf8");
  const body = md.render(rewriteMarkdown(raw));
  const content = `
    <p class="mb-4 text-sm text-muted">From <code class="rounded bg-black/5 px-1">docs/${page.source}</code></p>
    <article class="prose-cicerone" aria-label="${page.title}">
      ${body}
    </article>
    <p class="mt-10 border-t border-line pt-4 text-sm text-muted">
      Source:
      <a href="${githubBlob}/docs/${page.source}" rel="noopener noreferrer">docs/${page.source}<span class="sr-only"> on GitHub</span></a>
    </p>
  `;
  const html = renderLayout({
    title: page.title,
    description: page.description,
    content,
    root: "../",
    canonical: `${siteOrigin}${page.path}`,
    ogType: "article",
    nav: page.nav,
    structuredData: {
      "@context": "https://schema.org",
      "@type": "TechArticle",
      headline: page.title,
      description: page.description,
      url: `${siteOrigin}${page.path}`,
      isPartOf: { "@type": "WebSite", name: "Cicerone", url: siteOrigin },
      author: { "@type": "Organization", name: "torbido-hq", url: "https://github.com/torbido-hq" },
    },
  });
  writeFileSync(join(distDocs, page.out), html);
  return page;
}

function buildDocsIndex(pages) {
  const cards = pages
    .map(
      (p) => `
      <li class="border-b border-line py-5">
        <a class="text-lg font-semibold text-ink no-underline hover:text-cyan hover:underline" href="${p.out}">${p.title}</a>
        <p class="mt-1 text-ink-soft">${p.description}</p>
        <p class="mt-1 font-mono text-sm text-muted">docs/${p.source}</p>
      </li>`,
    )
    .join("\n");

  const content = `
    <h1 class="mb-3 font-sans text-page text-ink">Documentation</h1>
    <p class="mb-8 max-w-[40rem] text-ink-soft">
      Rendered from markdown under <code class="rounded bg-black/5 px-1">docs/</code>.
      Edit those files; this site rebuilds them on deploy.
    </p>
    <nav aria-label="Documentation pages">
      <ul class="list-none space-y-6 p-0 border-t border-line">${cards}</ul>
    </nav>
    <p class="mt-10 text-sm text-muted">
      Serve OpenAPI:
      <a href="openapi/serve.openapi.json">docs/openapi/serve.openapi.json</a>
    </p>
  `;

  writeFileSync(
    join(distDocs, "index.html"),
    renderLayout({
      title: "Documentation",
      description:
        "Cicerone documentation: tutorial and architecture, rendered from docs/ markdown.",
      content,
      root: "../",
      canonical: `${siteOrigin}/docs/`,
      ogType: "website",
      nav: "docs",
      structuredData: {
        "@context": "https://schema.org",
        "@type": "CollectionPage",
        name: "Cicerone documentation",
        description:
          "Tutorial and architecture docs for the Cicerone batch recommender.",
        url: `${siteOrigin}/docs/`,
        isPartOf: { "@type": "WebSite", name: "Cicerone", url: siteOrigin },
      },
    }),
  );
}

function writeSitemap(pages) {
  const urls = [
    { loc: `${siteOrigin}/`, priority: "1.0" },
    { loc: `${siteOrigin}/docs/`, priority: "0.9" },
    ...pages.map((p) => ({ loc: `${siteOrigin}${p.path}`, priority: "0.8" })),
  ];
  const body = urls
    .map(
      (u) => `  <url>
    <loc>${u.loc}</loc>
    <changefreq>weekly</changefreq>
    <priority>${u.priority}</priority>
  </url>`,
    )
    .join("\n");
  writeFileSync(
    join(distRoot, "sitemap.xml"),
    `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${body}
</urlset>
`,
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
  writeSitemap(built);
  console.log(`Rendered ${built.length} docs pages → dist/docs/ (+ sitemap.xml)`);
}

main();

#!/usr/bin/env node
/**
 * Sync repo docs/*.md into Starlight content with frontmatter.
 * docs/ remains the source of truth; run before `astro build` / `astro dev`.
 */
import { mkdirSync, readFileSync, writeFileSync, copyFileSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const websiteRoot = resolve(__dirname, "..");
const repoRoot = resolve(websiteRoot, "..");
const docsSrc = join(repoRoot, "docs");
const outDir = join(websiteRoot, "src/content/docs");
const githubBlob = "https://github.com/torbido-hq/cicerone/blob/main";

const PAGES = [
  {
    source: "tutorial.md",
    out: "tutorial.md",
    title: "Tutorial",
    description:
      "Hands-on Cicerone walkthrough: sample data, batch job, serve API, and dashboard.",
  },
  {
    source: "architecture.md",
    out: "architecture.md",
    title: "Architecture",
    description:
      "How Cicerone packages fit together: I/O, model strategies, job loop, serve, and dashboard.",
  },
];

function rewrite(source) {
  let text = source;

  // Drop the repo logo banner — Starlight already brands the chrome.
  text = text.replace(
    /^(?:\s*!\[[^\]]*\]\([^)]*cicerone-logo[^)]*\)\s*\n+)+/i,
    "",
  );
  text = text.replace(
    /^\s*(?:<picture>[\s\S]*?<\/picture>|<img\s+[^>]*cicerone-logo[^>]*>)\s*\n+/i,
    "",
  );

  // Strip a leading H1 — Starlight already renders `title` as the page heading.
  text = text.replace(/^#\s+[^\n]+\n+/, "");

  text = text.replace(/\]\(\.\.\/([^)#]+)(#[^)]*)?\)/g, (_m, path, hash = "") => {
    return `](${githubBlob}/${path}${hash})`;
  });

  text = text.replace(/\]\(([^)/]+)\.md(#[^)]*)?\)/g, (_m, name, hash = "") => {
    return `](/${name}/${hash || ""})`;
  });

  text = text.replace(/\.\.\/src\/cicerone\/static\//g, "/");
  text = text.replace(/\]\(images\//g, "](/images/");
  text = text.replace(/src="images\//g, 'src="/images/');

  return text.trim() + "\n";
}

mkdirSync(outDir, { recursive: true });

for (const page of PAGES) {
  const raw = readFileSync(join(docsSrc, page.source), "utf8");
  const body = rewrite(raw);
  const frontmatter = `---
title: ${JSON.stringify(page.title)}
description: ${JSON.stringify(page.description)}
editUrl: ${JSON.stringify(`${githubBlob}/docs/${page.source}`)}
---

`;
  writeFileSync(join(outDir, page.out), frontmatter + body);
  console.log(`synced docs/${page.source} → src/content/docs/${page.out}`);
}

const openapiSrc = join(docsSrc, "openapi/serve.openapi.json");
const openapiDst = join(websiteRoot, "public/openapi/serve.openapi.json");
if (existsSync(openapiSrc)) {
  mkdirSync(dirname(openapiDst), { recursive: true });
  copyFileSync(openapiSrc, openapiDst);
}

#!/usr/bin/env node
/**
 * Sync repo docs/*.md into Starlight content with frontmatter.
 * docs/ remains the source of truth; run before `astro build` / `astro dev`.
 */
import {
  mkdirSync,
  readFileSync,
  writeFileSync,
  copyFileSync,
  existsSync,
  readdirSync,
  unlinkSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseLatestRelease } from "../src/lib/changelog.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const websiteRoot = resolve(__dirname, "..");
const repoRoot = resolve(websiteRoot, "..");
const docsSrc = join(repoRoot, "docs");
const outDir = join(websiteRoot, "src/content/docs");
const githubBlob = "https://github.com/torbido-hq/cicerone/blob/main";

const PAGES = [
  {
    source: "how-it-works.md",
    out: "how-it-works.md",
    title: "How it works",
    description:
      "Batch pipeline, recommendation strategies, combiners, and how they differ.",
  },
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
  {
    source: "incremental-events.md",
    out: "incremental-events.md",
    title: "Incremental events",
    description:
      "EventSource backends, micro-batch incremental updates, and write-through to serve.",
  },
  {
    source: "experiments.md",
    out: "experiments.md",
    title: "Experiments",
    description:
      "Sticky A/B tests of ranking recipes, sequential stats, guardrails, and promote.",
  },
];

function rewrite(source) {
  let text = source;

  text = text.replace(
    /^\s*<img\s+src="(?:\.\.\/src\/cicerone\/static\/cicerone-logo\.svg|images\/cicerone-logo\.svg)"[^>]*>\s*\n+/i,
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
  text = text.replace(/\]\(images\//g, "](/images/docs/");
  text = text.replace(/src="images\//g, 'src="/images/docs/');

  return text.trim() + "\n";
}

mkdirSync(outDir, { recursive: true });

for (const page of PAGES) {
  const raw = readFileSync(join(docsSrc, page.source), "utf8");
  const body = rewrite(raw);
  const frontmatter = `---
title: ${JSON.stringify(page.title)}
description: ${JSON.stringify(page.description)}
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

const imagesSrc = join(docsSrc, "images");
const imagesDst = join(websiteRoot, "public/images/docs");
if (existsSync(imagesSrc)) {
  mkdirSync(imagesDst, { recursive: true });
  const syncedImages = new Set();
  for (const entry of readdirSync(imagesSrc, { withFileTypes: true })) {
    if (!entry.isFile()) continue;
    copyFileSync(join(imagesSrc, entry.name), join(imagesDst, entry.name));
    syncedImages.add(entry.name);
    console.log(`synced docs/images/${entry.name} → public/images/docs/${entry.name}`);
  }
  for (const entry of readdirSync(imagesDst, { withFileTypes: true })) {
    if (!entry.isFile() || syncedImages.has(entry.name)) continue;
    unlinkSync(join(imagesDst, entry.name));
    console.log(`removed stale public/images/docs/${entry.name}`);
  }
}

const changelogSrc = join(repoRoot, "CHANGELOG.md");
const latestReleaseOut = join(websiteRoot, "src/generated/latest-release.json");
mkdirSync(dirname(latestReleaseOut), { recursive: true });
const latestRelease = existsSync(changelogSrc)
  ? parseLatestRelease(readFileSync(changelogSrc, "utf8"))
  : null;
writeFileSync(latestReleaseOut, `${JSON.stringify(latestRelease)}\n`);
console.log(`synced CHANGELOG.md → src/generated/latest-release.json`);

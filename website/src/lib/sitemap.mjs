import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

import { articleHrefFromFilename, parseFrontmatter } from './articles.mjs';

const ARTICLE_EXT = /\.(mdx?|markdown)$/i;

export const OPENAPI_SITEMAP_URL = 'https://cicerone.dev/openapi/';

export const DOC_SITEMAP_PAGES = Object.freeze([
	['how-it-works.md', '/how-it-works/'],
	['tutorial.md', '/tutorial/'],
	['architecture.md', '/architecture/'],
	['incremental-events.md', '/incremental-events/'],
	['experiments.md', '/experiments/'],
]);

export function frontmatterLastmod(fm) {
	if (fm == null || typeof fm !== 'object') return undefined;
	for (const value of [fm.lastUpdated, fm.date]) {
		if (value instanceof Date && !Number.isNaN(value.getTime())) return value;
	}
	return undefined;
}

export function articlePathFromFilename(name) {
	return articleHrefFromFilename(name);
}

export function articleSitemapLastmods(articlesDir, { site = 'https://cicerone.dev' } = {}) {
	const lastmods = new Map();
	if (!existsSync(articlesDir)) return lastmods;
	for (const entry of readdirSync(articlesDir, { withFileTypes: true })) {
		if (!entry.isFile()) continue;
		if (entry.name.startsWith('_') || entry.name.startsWith('.')) continue;
		if (!ARTICLE_EXT.test(entry.name)) continue;
		const raw = readFileSync(join(articlesDir, entry.name), 'utf8');
		const lastmod = frontmatterLastmod(parseFrontmatter(raw));
		if (!lastmod) continue;
		lastmods.set(new URL(articlePathFromFilename(entry.name), site).href, lastmod);
	}
	return lastmods;
}

export function applySitemapLastmod(item, lastmods) {
	if (item == null || typeof item !== 'object') return item;
	const lastmod = lastmods.get(item.url);
	if (!lastmod) return item;
	return { ...item, lastmod };
}

export function mergeLastmods(...maps) {
	const lastmods = new Map();
	for (const map of maps) {
		if (!map) continue;
		for (const [url, date] of map) lastmods.set(url, date);
	}
	return lastmods;
}

export function docsSitemapLastmods(docsDir, { site = 'https://cicerone.dev' } = {}) {
	const lastmods = new Map();
	if (!existsSync(docsDir)) return lastmods;
	for (const [name, path] of DOC_SITEMAP_PAGES) {
		const file = join(docsDir, name);
		if (!existsSync(file)) continue;
		lastmods.set(new URL(path, site).href, statSync(file).mtime);
	}
	return lastmods;
}

import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import { ARTICLES_PREFIX, parseFrontmatter } from './articles.mjs';

const ARTICLE_EXT = /\.(mdx?|markdown)$/i;

export function frontmatterLastmod(fm) {
	if (fm == null || typeof fm !== 'object') return undefined;
	for (const value of [fm.lastUpdated, fm.date]) {
		if (value instanceof Date && !Number.isNaN(value.getTime())) return value;
	}
	return undefined;
}

export function articlePathFromFilename(name) {
	const stem = String(name).replace(ARTICLE_EXT, '');
	return `/${ARTICLES_PREFIX}/${stem}/`;
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

import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parse as parseYaml } from 'yaml';

const ARTICLE_EXT = /\.(mdx?|markdown)$/i;
export const ARTICLES_PREFIX = 'articles';
export const HOME_ARTICLE_LIMIT = 6;

// One Astro key: slash variants are one static route (duplicate keys warn,
// later a hard error). GitHub Pages serves dir/index.html at /path and /path/.
export const articleRedirects = {
	'/articles/welcome-to-your-own-recommender/': '/articles/a-nightly-table-next-to-your-orders/',
};

export function articlesContentDir(websiteRoot) {
	return join(websiteRoot, 'src/content/docs', ARTICLES_PREFIX);
}

export function defaultArticlesDir() {
	const fromModule = articlesContentDir(join(dirname(fileURLToPath(import.meta.url)), '../..'));
	if (existsSync(fromModule)) return fromModule;
	return articlesContentDir(process.cwd());
}

export function articleHrefFromFilename(name) {
	const stem = String(name).replace(ARTICLE_EXT, '');
	return `/${ARTICLES_PREFIX}/${stem}/`;
}

function isYamlTruthy(value) {
	if (value === true) return true;
	if (typeof value === 'string') return /^(?:true|yes|on)$/i.test(value.trim());
	return false;
}

export function parseFrontmatter(raw) {
	const text = String(raw).replace(/^\uFEFF/, '');
	const match = /^---[ \t]*\r?\n([\s\S]*?)\r?\n---(?:[ \t]*\r?\n|$)/.exec(text);
	if (!match) return {};
	try {
		const parsed = parseYaml(match[1], { version: '1.1' });
		if (parsed == null || typeof parsed !== 'object' || Array.isArray(parsed)) return {};
		return parsed;
	} catch {
		return {};
	}
}

export function articleIsVisible(raw, { production = process.env.NODE_ENV === 'production' } = {}) {
	const fm = parseFrontmatter(raw);
	if (production && isYamlTruthy(fm.draft)) return false;
	return true;
}

export function hasPublishedArticles(articlesDir, { production = process.env.NODE_ENV === 'production' } = {}) {
	return listPublishedArticles(articlesDir, { production }).length > 0;
}

function isoDay(value) {
	if (value instanceof Date && !Number.isNaN(value.getTime())) {
		return value.toISOString().slice(0, 10);
	}
	if (typeof value === 'string') {
		const match = /^(\d{4}-\d{2}-\d{2})/.exec(value.trim());
		if (match) return match[1];
	}
	return '';
}

function listingBlurb(fm) {
	for (const key of ['excerpt', 'description']) {
		const value = fm[key];
		if (typeof value === 'string' && value.trim()) return value.trim();
	}
	return '';
}

function articleFiles(articlesDir) {
	if (!existsSync(articlesDir)) return [];
	return readdirSync(articlesDir, { withFileTypes: true }).filter((entry) => {
		if (!entry.isFile()) return false;
		if (entry.name.startsWith('_') || entry.name.startsWith('.')) return false;
		return ARTICLE_EXT.test(entry.name);
	});
}

/** Newest `date` first. */
export function listPublishedArticles(
	articlesDir,
	{ production = process.env.NODE_ENV === 'production', limit } = {},
) {
	const items = [];
	for (const entry of articleFiles(articlesDir)) {
		const raw = readFileSync(join(articlesDir, entry.name), 'utf8');
		if (!articleIsVisible(raw, { production })) continue;
		const fm = parseFrontmatter(raw);
		const title = typeof fm.title === 'string' ? fm.title.trim() : '';
		if (!title) continue;
		items.push({
			href: articleHrefFromFilename(entry.name),
			title,
			date: isoDay(fm.date),
			excerpt: listingBlurb(fm),
		});
	}
	items.sort((a, b) => {
		if (a.date !== b.date) return a.date < b.date ? 1 : -1;
		return a.title.localeCompare(b.title);
	});
	if (Number.isInteger(limit) && limit >= 0) return items.slice(0, limit);
	return items;
}

export function articlesSidebarGroup(articlesDir, { production = process.env.NODE_ENV === 'production' } = {}) {
	const posts = listPublishedArticles(articlesDir, { production });
	if (posts.length === 0) return undefined;
	return {
		label: 'Articles',
		items: [
			{ label: 'All articles', link: `/${ARTICLES_PREFIX}/` },
			...posts.map((post) => ({ label: post.title, link: post.href })),
		],
	};
}

/** Listing vs post; shapes from starlight-blog `libs/page.ts` route helpers. */
export function articlesLayoutKind(id) {
	if (typeof id !== 'string' || !id) return undefined;
	const slug = id.replace(/\/+$/, '');
	if (!slug) return undefined;
	if (slug === ARTICLES_PREFIX) return 'index';
	const listing = new RegExp(`^${ARTICLES_PREFIX}/(?:\\d+|tags/.+|authors/.+)$`);
	if (listing.test(slug)) return 'index';
	if (slug.startsWith(`${ARTICLES_PREFIX}/`)) return 'post';
	return undefined;
}

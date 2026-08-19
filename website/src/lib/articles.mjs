import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parse as parseYaml } from 'yaml';

const ARTICLE_EXT = /\.(mdx?|markdown)$/i;
export const ARTICLES_PREFIX = 'articles';

export function articlesContentDir(websiteRoot) {
	return join(websiteRoot, 'src/content/docs', ARTICLES_PREFIX);
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
	if (!existsSync(articlesDir)) return false;
	for (const entry of readdirSync(articlesDir, { withFileTypes: true })) {
		if (!entry.isFile()) continue;
		if (entry.name.startsWith('_') || entry.name.startsWith('.')) continue;
		if (!ARTICLE_EXT.test(entry.name)) continue;
		const raw = readFileSync(join(articlesDir, entry.name), 'utf8');
		if (articleIsVisible(raw, { production })) return true;
	}
	return false;
}

/** `index` = listing / tags / authors; `post` = an article body. */
export function articlesLayoutKind(id) {
	if (id === ARTICLES_PREFIX) return 'index';
	if (!id.startsWith(`${ARTICLES_PREFIX}/`)) return undefined;
	const rest = id.slice(ARTICLES_PREFIX.length + 1);
	if (/^\d+$/.test(rest) || rest.startsWith('tags/') || rest.startsWith('authors/')) return 'index';
	return 'post';
}

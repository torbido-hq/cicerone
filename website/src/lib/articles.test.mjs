import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';

import {
	ARTICLES_PREFIX,
	HOME_ARTICLE_LIMIT,
	articleHrefFromFilename,
	articlesSidebarGroup,
	articleIsVisible,
	articleRedirects,
	articlesContentDir,
	articlesLayoutKind,
	defaultArticlesDir,
	hasPublishedArticles,
	listPublishedArticles,
	parseFrontmatter,
} from './articles.mjs';

test('parseFrontmatter reads draft regardless of field order', () => {
	const fm = parseFrontmatter(`---
title: Hello
authors:
  - nicholas
draft: true
date: 2026-08-19
---
Body
`);
	assert.equal(fm.draft, true);
	assert.equal(fm.title, 'Hello');
});

test('parseFrontmatter treats YAML 1.1 yes/on as true', () => {
	assert.equal(parseFrontmatter('---\ndraft: yes\n---\n').draft, true);
	assert.equal(parseFrontmatter('---\ndraft: on\n---\n').draft, true);
});

test('parseFrontmatter keeps quoted true as a string', () => {
	assert.equal(parseFrontmatter('---\ndraft: "true"\n---\n').draft, 'true');
});

test('parseFrontmatter returns {} for missing or invalid YAML', () => {
	assert.deepEqual(parseFrontmatter('# no frontmatter\n'), {});
	assert.deepEqual(parseFrontmatter('---\n[unterminated\n---\n'), {});
});

test('articleIsVisible hides drafts only in production', () => {
	const draft = '---\ntitle: x\ndraft: true\n---\n';
	const quoted = '---\ndraft: "yes"\n---\n';
	assert.equal(articleIsVisible(draft, { production: true }), false);
	assert.equal(articleIsVisible(draft, { production: false }), true);
	assert.equal(articleIsVisible(quoted, { production: true }), false);
	assert.equal(articleIsVisible('---\ndraft: false\n---\n', { production: true }), true);
});

test('hasPublishedArticles skips drafts, dotfiles, and non-markdown', () => {
	const dir = mkdtempSync(join(tmpdir(), 'cicerone-articles-'));
	writeFileSync(join(dir, '_skip.md'), '---\ntitle: x\n---\n');
	writeFileSync(join(dir, 'notes.txt'), '---\ntitle: x\n---\n');
	writeFileSync(join(dir, 'wip.md'), '---\ntitle: x\ndraft: true\n---\n');
	assert.equal(hasPublishedArticles(dir, { production: true }), false);
	writeFileSync(join(dir, 'live.md'), '---\ntitle: live\n---\n');
	assert.equal(hasPublishedArticles(dir, { production: true }), true);
});

test('articlesLayoutKind classifies listing vs post from the route id', () => {
	assert.equal(articlesLayoutKind(ARTICLES_PREFIX), 'index');
	assert.equal(articlesLayoutKind(`${ARTICLES_PREFIX}/2`), 'index');
	assert.equal(articlesLayoutKind(`${ARTICLES_PREFIX}/2/`), 'index');
	assert.equal(articlesLayoutKind(`${ARTICLES_PREFIX}/tags/foo`), 'index');
	assert.equal(articlesLayoutKind(`${ARTICLES_PREFIX}/tags/foo/`), 'index');
	assert.equal(articlesLayoutKind(`${ARTICLES_PREFIX}/authors/nicholas-wieland`), 'index');
	assert.equal(articlesLayoutKind(`${ARTICLES_PREFIX}/hello-world`), 'post');
	assert.equal(articlesLayoutKind(`${ARTICLES_PREFIX}/hello-world/`), 'post');
	assert.equal(articlesLayoutKind('tutorial'), undefined);
	assert.equal(articlesLayoutKind(undefined), undefined);
	assert.equal(articlesLayoutKind(''), undefined);
});

test('articlesContentDir nests under the shared prefix', () => {
	assert.equal(articlesContentDir('/site'), `/site/src/content/docs/${ARTICLES_PREFIX}`);
});

test('defaultArticlesDir is the site articles collection', () => {
	assert.equal(
		existsSync(join(defaultArticlesDir(), 'a-nightly-table-next-to-your-orders.md')),
		true,
	);
});

test('articleHrefFromFilename is the public articles URL', () => {
	assert.equal(
		articleHrefFromFilename('a-nightly-table-next-to-your-orders.md'),
		'/articles/a-nightly-table-next-to-your-orders/',
	);
});

test('listPublishedArticles is a newest-first feed from frontmatter', () => {
	const dir = mkdtempSync(join(tmpdir(), 'cicerone-articles-feed-'));
	writeFileSync(
		join(dir, 'older.md'),
		'---\ntitle: Older\ndate: 2026-08-20\nexcerpt: First published.\n---\n',
	);
	writeFileSync(
		join(dir, 'newer.md'),
		'---\ntitle: Newer\ndate: 2026-08-28\ndescription: Falls back when excerpt is missing.\n---\n',
	);
	writeFileSync(join(dir, 'wip.md'), '---\ntitle: Draft\ndate: 2026-08-29\ndraft: true\n---\n');
	writeFileSync(join(dir, '_skip.md'), '---\ntitle: Skip\ndate: 2026-08-19\n---\n');
	const posts = listPublishedArticles(dir, { production: true });
	assert.deepEqual(
		posts.map((post) => post.title),
		['Newer', 'Older'],
	);
	assert.deepEqual(posts[0], {
		href: '/articles/newer/',
		title: 'Newer',
		date: '2026-08-28',
		excerpt: 'Falls back when excerpt is missing.',
	});
	assert.equal(posts[1].excerpt, 'First published.');
	assert.equal(listPublishedArticles(dir, { production: false }).length, 3);
	assert.equal(HOME_ARTICLE_LIMIT, 6);
	for (let i = 1; i <= 8; i += 1) {
		writeFileSync(
			join(dir, `post-${String(i).padStart(2, '0')}.md`),
			`---\ntitle: Post ${i}\ndate: 2026-07-${String(i).padStart(2, '0')}\n---\n`,
		);
	}
	const home = listPublishedArticles(dir, { production: true, limit: HOME_ARTICLE_LIMIT });
	assert.equal(home.length, HOME_ARTICLE_LIMIT);
	assert.deepEqual(
		home.map((post) => post.title),
		['Newer', 'Older', 'Post 8', 'Post 7', 'Post 6', 'Post 5'],
	);
	const nav = articlesSidebarGroup(dir, { production: true });
	assert.equal(nav?.label, 'Articles');
	assert.deepEqual(nav?.items[0], { label: 'All articles', link: '/articles/' });
	assert.equal(nav?.items.length, 1 + listPublishedArticles(dir, { production: true }).length);
	assert.equal(nav?.items[1]?.label, 'Newer');
});

test('old article redirect is one directory route covering both slash variants', () => {
	const froms = Object.keys(articleRedirects);
	assert.equal(froms.length, 1, 'duplicate slash keys collide in Astro');
	assert.deepEqual(froms, ['/articles/welcome-to-your-own-recommender/']);
	assert.equal(articleRedirects[froms[0]], '/articles/a-nightly-table-next-to-your-orders/');
});

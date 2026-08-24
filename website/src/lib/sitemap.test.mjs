import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';

import {
	applySitemapLastmod,
	articlePathFromFilename,
	articleSitemapLastmods,
	frontmatterLastmod,
} from './sitemap.mjs';

test('frontmatterLastmod prefers lastUpdated over date', () => {
	const published = new Date('2026-08-20T00:00:00.000Z');
	const updated = new Date('2026-08-24T00:00:00.000Z');
	assert.equal(frontmatterLastmod({ date: published, lastUpdated: updated }), updated);
	assert.equal(frontmatterLastmod({ date: published, lastUpdated: true }), published);
	assert.equal(frontmatterLastmod({ date: published }), published);
	assert.equal(frontmatterLastmod({}), undefined);
});

test('articlePathFromFilename is the public articles URL', () => {
	assert.equal(
		articlePathFromFilename('a-nightly-table-next-to-your-orders.md'),
		'/articles/a-nightly-table-next-to-your-orders/',
	);
});

test('articleSitemapLastmods maps the article URL to lastUpdated', () => {
	const dir = mkdtempSync(join(tmpdir(), 'cicerone-sitemap-'));
	writeFileSync(
		join(dir, 'a-nightly-table-next-to-your-orders.md'),
		`---
title: A nightly table next to your orders
date: 2026-08-20
lastUpdated: 2026-08-24
---
Body
`,
	);
	const lastmods = articleSitemapLastmods(dir);
	const lastmod = lastmods.get('https://cicerone.dev/articles/a-nightly-table-next-to-your-orders/');
	assert.ok(lastmod instanceof Date);
	assert.equal(lastmod.toISOString(), '2026-08-24T00:00:00.000Z');
});

test('applySitemapLastmod copies lastmod onto a matching item', () => {
	const lastmod = new Date('2026-08-24T00:00:00.000Z');
	const url = 'https://cicerone.dev/articles/a-nightly-table-next-to-your-orders/';
	const lastmods = new Map([[url, lastmod]]);
	assert.deepEqual(applySitemapLastmod({ url, changefreq: 'weekly' }, lastmods), {
		url,
		changefreq: 'weekly',
		lastmod,
	});
	assert.deepEqual(applySitemapLastmod({ url: 'https://cicerone.dev/tutorial/' }, lastmods), {
		url: 'https://cicerone.dev/tutorial/',
	});
});

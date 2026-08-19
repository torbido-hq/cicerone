import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';

import {
	ARTICLES_PREFIX,
	articleIsVisible,
	articlesContentDir,
	articlesLayoutKind,
	hasPublishedArticles,
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
	assert.equal(articlesLayoutKind(`${ARTICLES_PREFIX}/tags/foo`), 'index');
	assert.equal(articlesLayoutKind(`${ARTICLES_PREFIX}/authors/nicholas`), 'index');
	assert.equal(articlesLayoutKind(`${ARTICLES_PREFIX}/hello-world`), 'post');
	assert.equal(articlesLayoutKind('tutorial'), undefined);
});

test('articlesContentDir nests under the shared prefix', () => {
	assert.equal(articlesContentDir('/site'), `/site/src/content/docs/${ARTICLES_PREFIX}`);
});

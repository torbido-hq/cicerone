import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import {
	CHANGELOG_BLOB,
	changelogPath,
	githubHeadingSlug,
	latestReleaseFromRepo,
	parseLatestRelease,
} from './changelog.mjs';

test('githubHeadingSlug matches Keep a Changelog GitHub anchors', () => {
	assert.equal(githubHeadingSlug('[0.6.0] - 2026-08-20'), '060---2026-08-20');
	assert.equal(githubHeadingSlug('[1.1.0] - 2023-03-06'), '110---2023-03-06');
});

test('parseLatestRelease skips Unreleased and returns the first dated section', () => {
	const text = `# Changelog

## [Unreleased]

- pending

## [0.6.0] - 2026-08-20

- notes

## [0.5.1] - 2026-04-22

- older
`;
	assert.deepEqual(parseLatestRelease(text), {
		version: '0.6.0',
		date: '2026-08-20',
		url: `${CHANGELOG_BLOB}#060---2026-08-20`,
	});
});

test('parseLatestRelease returns null when no dated release exists', () => {
	assert.equal(parseLatestRelease('## [Unreleased]\n\n- wip\n'), null);
	assert.equal(parseLatestRelease(''), null);
});

test('latestReleaseFromRepo reads the repo CHANGELOG', () => {
	const fromFile = parseLatestRelease(readFileSync(changelogPath(), 'utf8'));
	assert.deepEqual(latestReleaseFromRepo(), fromFile);
	assert.ok(fromFile);
	assert.match(fromFile.version, /^\d+\.\d+\.\d+$/);
	assert.match(fromFile.url, /^https:\/\/github\.com\/torbido-hq\/cicerone\/blob\/main\/CHANGELOG\.md#/);
});

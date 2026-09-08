import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import {
	CHANGELOG_BLOB,
	PYPI_PROJECT,
	changelogPath,
	githubHeadingSlug,
	latestReleaseFromPypi,
	latestReleaseFromRepo,
	parseLatestRelease,
	parseReleaseForVersion,
} from './changelog.mjs';

test('githubHeadingSlug matches Keep a Changelog GitHub anchors', () => {
	assert.equal(githubHeadingSlug('[0.6.0] - 2026-08-20'), '060---2026-08-20');
	assert.equal(githubHeadingSlug('[1.1.0] - 2023-03-06'), '110---2023-03-06');
});

test('parseLatestRelease returns the first dated section', () => {
	const text = `# Changelog

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
	assert.equal(parseLatestRelease('## Notes\n\n- wip\n'), null);
	assert.equal(parseLatestRelease(''), null);
});

test('latestReleaseFromRepo reads the repo CHANGELOG', () => {
	const fromFile = parseLatestRelease(readFileSync(changelogPath(), 'utf8'));
	assert.deepEqual(latestReleaseFromRepo(), fromFile);
	assert.ok(fromFile);
	assert.match(fromFile.version, /^\d+\.\d+\.\d+$/);
	assert.match(fromFile.url, /^https:\/\/github\.com\/torbido-hq\/cicerone\/blob\/main\/CHANGELOG\.md#/);
});

test('parseReleaseForVersion uses the PyPI version, not the first CHANGELOG heading', () => {
	const text = `# Changelog

## [0.8.1] - 2026-09-09

- unreleased notes

## [0.8.0] - 2026-09-08

- shipped
`;
	assert.deepEqual(parseLatestRelease(text), {
		version: '0.8.1',
		date: '2026-09-09',
		url: `${CHANGELOG_BLOB}#081---2026-09-09`,
	});
	assert.deepEqual(parseReleaseForVersion(text, '0.8.0'), {
		version: '0.8.0',
		date: '2026-09-08',
		url: `${CHANGELOG_BLOB}#080---2026-09-08`,
	});
});

test('latestReleaseFromPypi ignores a newer unpublished CHANGELOG heading', () => {
	const text = `# Changelog

## [0.8.1] - 2026-09-09

- unreleased notes

## [0.8.0] - 2026-09-08

- shipped
`;
	const release = latestReleaseFromPypi({ info: { version: '0.8.0' } }, text);
	assert.deepEqual(release, {
		version: '0.8.0',
		date: '2026-09-08',
		url: `${CHANGELOG_BLOB}#080---2026-09-08`,
	});
});

test('latestReleaseFromPypi falls back to the PyPI upload date', () => {
	const release = latestReleaseFromPypi(
		{
			info: { version: '0.8.0' },
			releases: {
				'0.8.0': [{ upload_time_iso_8601: '2026-09-08T12:00:00.000000Z' }],
			},
		},
		'# Changelog\n',
	);
	assert.deepEqual(release, {
		version: '0.8.0',
		date: '2026-09-08',
		url: `https://pypi.org/project/${PYPI_PROJECT}/0.8.0/`,
	});
});

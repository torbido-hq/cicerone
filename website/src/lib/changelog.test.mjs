import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import {
	CHANGELOG_BLOB,
	PYPI_JSON_URL,
	PYPI_PROJECT,
	changelogPath,
	fetchPypiProject,
	githubHeadingSlug,
	latestReleaseFromPypi,
	latestReleaseFromRepo,
	parseLatestRelease,
	parseReleaseForVersion,
	resolveLatestRelease,
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

test('fetchPypiProject wraps invalid JSON as a clear Error', async () => {
	await assert.rejects(
		() =>
			fetchPypiProject(PYPI_JSON_URL, async () => ({
				ok: true,
				status: 200,
				json: async () => {
					throw new SyntaxError('Unexpected token');
				},
			})),
		{
			name: 'Error',
			message: `PyPI ${PYPI_JSON_URL} returned invalid JSON: Unexpected token`,
		},
	);
});

const PYPI_PROJECT_JSON = {
	info: { version: '0.8.0' },
	releases: {
		'0.8.0': [{ upload_time_iso_8601: '2026-09-08T12:00:00.000000Z' }],
	},
};

const CHANGELOG_TEXT = `# Changelog

## [0.7.0] - 2026-07-01

- shipped
`;

const PREVIOUS_RELEASE = {
	version: '0.8.0',
	date: '2026-09-08',
	url: `${CHANGELOG_BLOB}#080---2026-09-08`,
};

test('resolveLatestRelease uses PyPI when fetch succeeds', async () => {
	const resolved = await resolveLatestRelease({
		changelogText: CHANGELOG_TEXT,
		previous: PREVIOUS_RELEASE,
		fetchProject: async () => PYPI_PROJECT_JSON,
	});
	assert.deepEqual(resolved, {
		release: {
			version: '0.8.0',
			date: '2026-09-08',
			url: `https://pypi.org/project/${PYPI_PROJECT}/0.8.0/`,
		},
		stale: false,
		source: 'pypi',
		reason: null,
	});
});

test('resolveLatestRelease uses a previous valid file when PyPI fails', async () => {
	const resolved = await resolveLatestRelease({
		changelogText: CHANGELOG_TEXT,
		previous: PREVIOUS_RELEASE,
		fetchProject: async () => {
			throw new Error('PyPI https://pypi.org/pypi/cicerone-recommender/json returned 503');
		},
	});
	assert.deepEqual(resolved, {
		release: PREVIOUS_RELEASE,
		stale: true,
		source: 'previous',
		reason: 'PyPI https://pypi.org/pypi/cicerone-recommender/json returned 503',
	});
});

test('resolveLatestRelease falls back to CHANGELOG when PyPI fails and no previous file', async () => {
	const resolved = await resolveLatestRelease({
		changelogText: CHANGELOG_TEXT,
		previous: null,
		fetchProject: async () => {
			throw new Error('network down');
		},
	});
	assert.deepEqual(resolved, {
		release: {
			version: '0.7.0',
			date: '2026-07-01',
			url: `${CHANGELOG_BLOB}#070---2026-07-01`,
		},
		stale: true,
		source: 'changelog',
		reason: 'network down',
	});
});

test('resolveLatestRelease throws when PyPI fails and nothing else is available', async () => {
	await assert.rejects(
		() =>
			resolveLatestRelease({
				changelogText: '',
				previous: { version: 'not-a-release' },
				fetchProject: async () => {
					throw new Error('timed out');
				},
			}),
		{
			name: 'Error',
			message: 'Could not resolve latest release: timed out',
		},
	);
});

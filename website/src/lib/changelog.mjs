import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const CHANGELOG_BLOB =
	'https://github.com/torbido-hq/cicerone/blob/main/CHANGELOG.md';

const DATED_RELEASE = /^## \[(\d+\.\d+\.\d+)\] - (\d{4}-\d{2}-\d{2})\s*$/m;

/** GitHub heading slug for Keep a Changelog `## [x.y.z] - YYYY-MM-DD`. */
export function githubHeadingSlug(heading) {
	return heading
		.toLowerCase()
		.replace(/[^\p{L}\p{M}\p{Nd}\p{Pc}\s-]+/gu, '')
		.trim()
		.replace(/\s+/g, '-');
}

export function parseLatestRelease(text) {
	const match = DATED_RELEASE.exec(String(text));
	if (!match) return null;
	const version = match[1];
	const date = match[2];
	const slug = githubHeadingSlug(`[${version}] - ${date}`);
	return {
		version,
		date,
		url: `${CHANGELOG_BLOB}#${slug}`,
	};
}

export function changelogPath(fromUrl = import.meta.url) {
	return join(dirname(fileURLToPath(fromUrl)), '../../..', 'CHANGELOG.md');
}

export function latestReleaseFromRepo(fromUrl = import.meta.url) {
	return parseLatestRelease(readFileSync(changelogPath(fromUrl), 'utf8'));
}

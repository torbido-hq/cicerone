import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const CHANGELOG_BLOB =
	'https://github.com/torbido-hq/cicerone/blob/main/CHANGELOG.md';

export const PYPI_PROJECT = 'cicerone-recommender';
export const PYPI_JSON_URL = `https://pypi.org/pypi/${PYPI_PROJECT}/json`;

const DATED_RELEASE = /^## \[(\d+\.\d+\.\d+)\] - (\d{4}-\d{2}-\d{2})\s*$/gm;

/** GitHub heading slug for Keep a Changelog `## [x.y.z] - YYYY-MM-DD`. */
export function githubHeadingSlug(heading) {
	return heading
		.toLowerCase()
		.replace(/[^\p{L}\p{M}\p{Nd}\p{Pc}\s-]+/gu, '')
		.trim()
		.replace(/\s+/g, '-');
}

function releaseRecord(version, date) {
	const slug = githubHeadingSlug(`[${version}] - ${date}`);
	return {
		version,
		date,
		url: `${CHANGELOG_BLOB}#${slug}`,
	};
}

export function parseDatedReleases(text) {
	const out = [];
	const source = String(text);
	for (const match of source.matchAll(DATED_RELEASE)) {
		out.push(releaseRecord(match[1], match[2]));
	}
	return out;
}

export function parseLatestRelease(text) {
	return parseDatedReleases(text)[0] ?? null;
}

export function parseReleaseForVersion(text, version) {
	return parseDatedReleases(text).find((item) => item.version === version) ?? null;
}

export function pypiUploadDate(project, version) {
	const files = project?.releases?.[version];
	if (!Array.isArray(files) || files.length === 0) return null;
	const stamps = files
		.map((file) => file.upload_time_iso_8601 || file.upload_time)
		.filter((value) => typeof value === 'string' && value.length >= 10)
		.map((value) => value.slice(0, 10))
		.sort();
	return stamps[0] ?? null;
}

export function latestReleaseFromPypi(project, changelogText) {
	const version = project?.info?.version;
	if (typeof version !== 'string' || !/^\d+\.\d+\.\d+$/.test(version)) {
		throw new Error('PyPI project JSON is missing info.version');
	}
	const fromChangelog = parseReleaseForVersion(changelogText ?? '', version);
	if (fromChangelog) return fromChangelog;
	const date = pypiUploadDate(project, version);
	if (!date) {
		throw new Error(`PyPI project JSON has no upload time for ${version}`);
	}
	return {
		version,
		date,
		url: `https://pypi.org/project/${PYPI_PROJECT}/${version}/`,
	};
}

export async function fetchPypiProject(url = PYPI_JSON_URL, fetchImpl = fetch) {
	const response = await fetchImpl(url);
	if (!response.ok) {
		throw new Error(`PyPI ${url} returned ${response.status}`);
	}
	return response.json();
}

export function changelogPath(fromUrl = import.meta.url) {
	return join(dirname(fileURLToPath(fromUrl)), '../../..', 'CHANGELOG.md');
}

export function latestReleaseFromRepo(fromUrl = import.meta.url) {
	return parseLatestRelease(readFileSync(changelogPath(fromUrl), 'utf8'));
}

export function latestArticleLastmod(lastmods) {
	let latest;
	for (const date of lastmods?.values?.() ?? []) {
		if (latest == null || date > latest) latest = date;
	}
	return latest;
}

function rfc822(date) {
	return date.toUTCString();
}

function escapeRegExp(value) {
	return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export function stampRssChannelLink(xml, href = 'https://cicerone.dev/articles/') {
	return String(xml).replace(/(<channel>\s*<title>[^<]*<\/title>\s*(?:<description>[^<]*<\/description>)?\s*<link>)[^<]*(<\/link>)/, `$1${href}$2`);
}

export function stampRssUpdated(xml, { lastBuildDate, itemUpdatedByLink, channelLink } = {}) {
	let out = stampRssChannelLink(String(xml), channelLink);
	if (lastBuildDate instanceof Date && !Number.isNaN(lastBuildDate.getTime())) {
		const tag = `<lastBuildDate>${rfc822(lastBuildDate)}</lastBuildDate>`;
		if (/<lastBuildDate>/.test(out)) {
			out = out.replace(/<lastBuildDate>[^<]*<\/lastBuildDate>/, tag);
		} else {
			out = out.replace(/<language>[^<]*<\/language>/, (m) => `${m}${tag}`);
		}
	}
	for (const [link, updated] of itemUpdatedByLink ?? []) {
		if (!(updated instanceof Date) || Number.isNaN(updated.getTime())) continue;
		const itemUpdated = `<atom:updated>${updated.toISOString()}</atom:updated>`;
		const itemRe = new RegExp(
			`(<link>${escapeRegExp(link)}</link>[\\s\\S]*?<pubDate>[^<]*</pubDate>)(?:<atom:updated>[^<]*</atom:updated>)?`,
		);
		out = out.replace(itemRe, `$1${itemUpdated}`);
	}
	return out;
}

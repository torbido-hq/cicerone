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
	const source = String(xml);
	const channelAt = source.search(/<channel\b/i);
	if (channelAt === -1) return source;
	const fromChannel = source.slice(channelAt);
	const itemAt = fromChannel.search(/<item[\s>/]/i);
	const closeAt = fromChannel.search(/<\/channel>/i);
	const headEnd = itemAt === -1 ? closeAt : itemAt;
	if (headEnd === -1) return source;
	const absEnd = channelAt + headEnd;
	const images = [];
	const head = source.slice(channelAt, absEnd).replace(/<image\b[^>]*>[\s\S]*?<\/image>/gi, (block) => {
		images.push(block);
		return `\0IMAGE${images.length - 1}\0`;
	});
	const rewritten = head.replace(/(?<![:\w])<link>[^<]*<\/link>/i, `<link>${href}</link>`);
	return `${source.slice(0, channelAt)}${rewritten.replace(/\0IMAGE(\d+)\0/g, (_, i) => images[Number(i)])}${source.slice(absEnd)}`;
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

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { latestArticleLastmod, stampRssChannelLink, stampRssUpdated } from './rss.mjs';

const FEED = `<?xml version="1.0"?><rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom"><channel><title>Articles</title><language>en</language><item><title>A nightly table next to your orders</title><link>https://cicerone.dev/articles/a-nightly-table-next-to-your-orders/</link><guid isPermaLink="true">https://cicerone.dev/articles/a-nightly-table-next-to-your-orders/</guid><description>excerpt</description><pubDate>Thu, 20 Aug 2026 00:00:00 GMT</pubDate><content:encoded>body</content:encoded></item></channel></rss>`;

test('latestArticleLastmod is the newest lastUpdated', () => {
	const older = new Date('2026-08-20T00:00:00.000Z');
	const newer = new Date('2026-08-24T00:00:00.000Z');
	assert.equal(
		latestArticleLastmod(
			new Map([
				['https://cicerone.dev/articles/old/', older],
				['https://cicerone.dev/articles/a-nightly-table-next-to-your-orders/', newer],
			]),
		),
		newer,
	);
	assert.equal(latestArticleLastmod(new Map()), undefined);
});

test('stampRssUpdated keeps pubDate and writes atom:updated plus lastBuildDate', () => {
	const updated = new Date('2026-08-24T00:00:00.000Z');
	const stamped = stampRssUpdated(FEED, {
		lastBuildDate: updated,
		itemUpdatedByLink: new Map([
			['https://cicerone.dev/articles/a-nightly-table-next-to-your-orders/', updated],
		]),
	});
	assert.match(stamped, /<pubDate>Thu, 20 Aug 2026 00:00:00 GMT<\/pubDate>/);
	assert.match(stamped, /<lastBuildDate>Mon, 24 Aug 2026 00:00:00 GMT<\/lastBuildDate>/);
	assert.match(stamped, /<atom:updated>2026-08-24T00:00:00.000Z<\/atom:updated>/);
	assert.equal(stamped.includes('<atom:updated>'), true);
	const again = stampRssUpdated(stamped, {
		lastBuildDate: updated,
		itemUpdatedByLink: new Map([
			['https://cicerone.dev/articles/a-nightly-table-next-to-your-orders/', updated],
		]),
	});
	assert.equal(again.match(/<atom:updated>/g)?.length, 1);
	assert.equal(again.match(/<lastBuildDate>/g)?.length, 1);
});

test('stampRssChannelLink rewrites the channel link only', () => {
	const feed = `<channel><title>Articles</title><description>x</description><link>https://cicerone.dev/</link><item><link>https://cicerone.dev/articles/a/</link></item></channel>`;
	const stamped = stampRssChannelLink(feed);
	assert.match(stamped, /<channel><title>Articles<\/title><description>x<\/description><link>https:\/\/cicerone.dev\/articles\/<\/link>/);
	assert.match(stamped, /<item><link>https:\/\/cicerone.dev\/articles\/a\/<\/link>/);
});

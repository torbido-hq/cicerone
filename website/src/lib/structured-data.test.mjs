import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
	ARTICLES_DESCRIPTION,
	AUTHOR_DESCRIPTION,
	DEFAULT_OG_IMAGE_ALT,
	PUBLISHER,
	articleImageForId,
	homeStructuredData,
	listingDescriptionForId,
	ogImageForId,
} from './structured-data.mjs';

test('homeStructuredData includes WebSite and SoftwareApplication', () => {
	const data = homeStructuredData();
	const types = data['@graph'].map((node) => node['@type']);
	assert.deepEqual(types, ['WebSite', 'SoftwareApplication', 'Organization']);
	assert.equal(data['@graph'][0].url, 'https://cicerone.dev/');
	assert.deepEqual(data['@graph'][2], { ...PUBLISHER, '@id': 'https://cicerone.dev/#org' });
	assert.equal(data['@graph'][2].logo.url, 'https://cicerone.dev/images/docs/cicerone-logo.svg');
});

test('ogImageForId pairs the Stripe post image with its own alt', () => {
	const afternoon = ogImageForId('articles/this-afternoons-checkout-can-move-the-row');
	assert.equal(afternoon.url, 'https://cicerone.dev/images/afternoon-checkout-architecture.png');
	assert.match(afternoon.alt, /Stripe checkout/i);
	assert.notEqual(afternoon.alt, DEFAULT_OG_IMAGE_ALT);
	assert.equal(
		articleImageForId('articles/a-nightly-table-next-to-your-orders'),
		'https://cicerone.dev/images/docs/dashboard.png',
	);
	assert.equal(ogImageForId('articles/a-nightly-table-next-to-your-orders').alt, DEFAULT_OG_IMAGE_ALT);
});

test('listingDescriptionForId covers the index and author pages', () => {
	assert.equal(listingDescriptionForId('articles'), ARTICLES_DESCRIPTION);
	assert.equal(listingDescriptionForId('articles/authors/nicholas-wieland'), AUTHOR_DESCRIPTION);
	assert.equal(listingDescriptionForId('tutorial'), undefined);
});

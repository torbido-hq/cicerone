import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
	ARTICLES_DESCRIPTION,
	AUTHOR_DESCRIPTION,
	articleImageForId,
	homeStructuredData,
	listingDescriptionForId,
} from './structured-data.mjs';

test('homeStructuredData includes WebSite and SoftwareApplication', () => {
	const data = homeStructuredData();
	const types = data['@graph'].map((node) => node['@type']);
	assert.deepEqual(types, ['WebSite', 'SoftwareApplication', 'Organization']);
	assert.equal(data['@graph'][0].url, 'https://cicerone.dev/');
});

test('articleImageForId uses the architecture shot for the Stripe post', () => {
	assert.equal(
		articleImageForId('articles/this-afternoons-checkout-can-move-the-row'),
		'https://cicerone.dev/images/afternoon-checkout-architecture.png',
	);
	assert.equal(
		articleImageForId('articles/a-nightly-table-next-to-your-orders'),
		'https://cicerone.dev/images/docs/dashboard.png',
	);
});

test('listingDescriptionForId covers the index and author pages', () => {
	assert.equal(listingDescriptionForId('articles'), ARTICLES_DESCRIPTION);
	assert.equal(listingDescriptionForId('articles/authors/nicholas-wieland'), AUTHOR_DESCRIPTION);
	assert.equal(listingDescriptionForId('tutorial'), undefined);
});

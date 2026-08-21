import assert from 'node:assert/strict';
import { test } from 'node:test';

import { CONSENT_DENIED } from './consent.mjs';
import {
	initConsentBanner,
	readStoredConsent,
	tabWrapTarget,
	updateGtagConsent,
	writeStoredConsent,
} from './consent-ui.mjs';

test('tabWrapTarget cycles first/last only', () => {
	const items = ['a', 'b', 'c'];
	assert.equal(tabWrapTarget(items, 'a', true), 'c');
	assert.equal(tabWrapTarget(items, 'c', false), 'a');
	assert.equal(tabWrapTarget(items, 'b', false), null);
	assert.equal(tabWrapTarget(items, 'b', true), null);
	assert.equal(tabWrapTarget([], 'a', false), null);
});

test('readStoredConsent and writeStoredConsent do not throw without localStorage', () => {
	assert.equal(readStoredConsent(), null);
	assert.doesNotThrow(() => writeStoredConsent(CONSENT_DENIED));
});

test('updateGtagConsent no-ops when gtag is missing', () => {
	assert.doesNotThrow(() => updateGtagConsent(CONSENT_DENIED));
});

test('initConsentBanner shows controls when gtag is missing', () => {
	const root = {
		hidden: true,
		setAttribute() {},
		addEventListener() {},
		querySelectorAll: () => [],
		focus() {},
	};
	const footer = { hidden: true };
	const doc = {
		getElementById: () => root,
		querySelector: () => footer,
		addEventListener() {},
		activeElement: null,
	};
	initConsentBanner(doc);
	assert.equal(footer.hidden, false);
	assert.equal(root.hidden, false);
});

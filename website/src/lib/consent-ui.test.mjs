import assert from 'node:assert/strict';
import { test } from 'node:test';

import { CONSENT_ANALYTICS, CONSENT_DENIED, CONSENT_STORAGE_KEY } from './consent.mjs';
import {
	analyticsStorageJustGranted,
	applyConsentState,
	initConsentBanner,
	readStoredConsent,
	sendGtagPageView,
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

test('sendGtagPageView no-ops when gtag is missing', () => {
	assert.doesNotThrow(() => sendGtagPageView());
});

test('analyticsStorageJustGranted is only true on the denied-to-granted edge', () => {
	assert.equal(analyticsStorageJustGranted(null, CONSENT_ANALYTICS), true);
	assert.equal(analyticsStorageJustGranted(CONSENT_DENIED, CONSENT_ANALYTICS), true);
	assert.equal(analyticsStorageJustGranted(CONSENT_ANALYTICS, CONSENT_ANALYTICS), false);
	assert.equal(analyticsStorageJustGranted(CONSENT_ANALYTICS, CONSENT_DENIED), false);
	assert.equal(analyticsStorageJustGranted(null, CONSENT_DENIED), false);
});

test('applyConsentState sends page_view only when analytics is newly granted', () => {
	const previous = globalThis.gtag;
	const store = new Map();
	const previousStorage = globalThis.localStorage;
	globalThis.localStorage = {
		getItem: (key) => (store.has(key) ? store.get(key) : null),
		setItem: (key, value) => {
			store.set(key, String(value));
		},
		removeItem: (key) => {
			store.delete(key);
		},
	};
	const calls = [];
	globalThis.gtag = (...args) => {
		calls.push(args);
	};
	try {
		applyConsentState({ ...CONSENT_DENIED });
		assert.deepEqual(calls, [['consent', 'update', CONSENT_DENIED]]);
		assert.equal(store.has(CONSENT_STORAGE_KEY), true);

		calls.length = 0;
		applyConsentState({ ...CONSENT_ANALYTICS });
		assert.deepEqual(calls, [
			['consent', 'update', CONSENT_ANALYTICS],
			['event', 'page_view'],
		]);

		calls.length = 0;
		applyConsentState({ ...CONSENT_ANALYTICS });
		assert.deepEqual(calls, [['consent', 'update', CONSENT_ANALYTICS]]);
	} finally {
		globalThis.gtag = previous;
		globalThis.localStorage = previousStorage;
	}
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

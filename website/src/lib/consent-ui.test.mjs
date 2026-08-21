import assert from 'node:assert/strict';
import { test } from 'node:test';

import { CONSENT_ANALYTICS, CONSENT_DENIED } from './consent.mjs';
import {
	analyticsStorageJustGranted,
	applyConsentState,
	initConsentBanner,
	loadGoogleTag,
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

test('loadGoogleTag no-ops when gtag is missing', () => {
	assert.doesNotThrow(() => loadGoogleTag());
});

test('analyticsStorageJustGranted is only true on the denied-to-granted edge', () => {
	assert.equal(analyticsStorageJustGranted(null, CONSENT_ANALYTICS), true);
	assert.equal(analyticsStorageJustGranted(CONSENT_DENIED, CONSENT_ANALYTICS), true);
	assert.equal(analyticsStorageJustGranted(CONSENT_ANALYTICS, CONSENT_ANALYTICS), false);
	assert.equal(analyticsStorageJustGranted(CONSENT_ANALYTICS, CONSENT_DENIED), false);
	assert.equal(analyticsStorageJustGranted(null, CONSENT_DENIED), false);
});

test('applyConsentState loads gtag only when analytics is newly granted', () => {
	const previousGtag = globalThis.gtag;
	const previousStorage = globalThis.localStorage;
	const previousLoaded = globalThis.__ciceroneGtagLoaded;
	const previousId = globalThis.__CICERONE_GA_ID;
	const previousDocument = globalThis.document;
	const store = new Map();
	const appended = [];
	globalThis.localStorage = {
		getItem: (key) => (store.has(key) ? store.get(key) : null),
		setItem: (key, value) => {
			store.set(key, String(value));
		},
		removeItem: (key) => {
			store.delete(key);
		},
	};
	globalThis.document = {
		createElement: (tag) => ({ tagName: tag, async: false, src: '' }),
		head: {
			appendChild: (node) => {
				appended.push(node);
				return node;
			},
		},
	};
	globalThis.__CICERONE_GA_ID = 'G-ABC123';
	globalThis.__ciceroneGtagLoaded = false;
	const calls = [];
	globalThis.gtag = (...args) => {
		calls.push(args);
	};
	try {
		applyConsentState({ ...CONSENT_DENIED });
		assert.deepEqual(calls, [['consent', 'update', CONSENT_DENIED]]);
		assert.equal(appended.length, 0);

		calls.length = 0;
		applyConsentState({ ...CONSENT_ANALYTICS });
		assert.equal(calls[0][0], 'consent');
		assert.equal(calls[1][0], 'js');
		assert.deepEqual(calls[2], ['config', 'G-ABC123']);
		assert.equal(appended.length, 1);
		assert.equal(appended[0].src, 'https://www.googletagmanager.com/gtag/js?id=G-ABC123');
		assert.equal(globalThis.__ciceroneGtagLoaded, true);

		calls.length = 0;
		applyConsentState({ ...CONSENT_ANALYTICS });
		assert.deepEqual(calls, [['consent', 'update', CONSENT_ANALYTICS]]);
	} finally {
		globalThis.gtag = previousGtag;
		globalThis.localStorage = previousStorage;
		globalThis.__ciceroneGtagLoaded = previousLoaded;
		globalThis.__CICERONE_GA_ID = previousId;
		globalThis.document = previousDocument;
	}
});

test('initConsentBanner loads gtag when stored analytics consent is granted', () => {
	const previousGtag = globalThis.gtag;
	const previousStorage = globalThis.localStorage;
	const previousLoaded = globalThis.__ciceroneGtagLoaded;
	const previousId = globalThis.__CICERONE_GA_ID;
	const previousDocument = globalThis.document;
	globalThis.localStorage = {
		getItem: () => JSON.stringify(CONSENT_ANALYTICS),
		setItem() {},
		removeItem() {},
	};
	globalThis.__CICERONE_GA_ID = 'G-ABC123';
	globalThis.__ciceroneGtagLoaded = false;
	const calls = [];
	globalThis.gtag = (...args) => {
		calls.push(args);
	};
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
		createElement: (tag) => ({ tagName: tag, async: false, src: '' }),
		head: { appendChild: (node) => node },
	};
	globalThis.document = doc;
	try {
		initConsentBanner(doc);
		assert.equal(root.hidden, true);
		assert.equal(calls[0][0], 'js');
		assert.deepEqual(calls[1], ['config', 'G-ABC123']);
		assert.equal(globalThis.__ciceroneGtagLoaded, true);
	} finally {
		globalThis.gtag = previousGtag;
		globalThis.localStorage = previousStorage;
		globalThis.__ciceroneGtagLoaded = previousLoaded;
		globalThis.__CICERONE_GA_ID = previousId;
		globalThis.document = previousDocument;
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

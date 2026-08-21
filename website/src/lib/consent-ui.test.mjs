import assert from 'node:assert/strict';
import { test } from 'node:test';

import { CONSENT_ANALYTICS, CONSENT_DENIED, CONSENT_STORAGE_KEY } from './consent.mjs';
import {
	analyticsStorageJustGranted,
	applyConsentState,
	initConsentBanner,
	loadGoogleTag,
	googleMeasurementId,
	readStoredConsent,
	tabWrapTarget,
	updateGtagConsent,
	writeStoredConsent,
} from './consent-ui.mjs';

const GTAG_GLOBAL_KEYS = [
	'gtag',
	'localStorage',
	'document',
	'__CICERONE_GA_ID',
	'__ciceroneGtagLoaded',
];

function memoryStorage(initial = {}) {
	const store = new Map(Object.entries(initial));
	return {
		getItem: (key) => (store.has(key) ? store.get(key) : null),
		setItem: (key, value) => {
			store.set(key, String(value));
		},
		removeItem: (key) => {
			store.delete(key);
		},
	};
}

function installGtagHarness({
	measurementId = 'G-ABC123',
	loaded = false,
	storage,
	document: doc,
} = {}) {
	const snapshot = Object.fromEntries(GTAG_GLOBAL_KEYS.map((key) => [key, globalThis[key]]));
	const calls = [];
	const appended = [];
	globalThis.__CICERONE_GA_ID = measurementId;
	globalThis.__ciceroneGtagLoaded = loaded;
	globalThis.gtag = (...args) => {
		calls.push(args);
	};
	if (storage) globalThis.localStorage = storage;
	globalThis.document =
		doc ??
		{
			createElement: (tag) => ({ tagName: tag, async: false, src: '' }),
			head: {
				appendChild: (node) => {
					appended.push(node);
					return node;
				},
			},
		};
	return {
		calls,
		appended,
		restore() {
			for (const key of GTAG_GLOBAL_KEYS) globalThis[key] = snapshot[key];
		},
	};
}

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

test('googleMeasurementId canonicalizes the runtime id once', () => {
	const harness = installGtagHarness({ measurementId: ' g-abc123 ' });
	try {
		assert.equal(googleMeasurementId(), 'G-ABC123');
	} finally {
		harness.restore();
	}
});

test('loadGoogleTag leaves unloaded when document cannot inject a script', () => {
	const harness = installGtagHarness({ document: { createElement: () => ({}) } });
	try {
		loadGoogleTag();
		assert.equal(globalThis.__ciceroneGtagLoaded, false);
		assert.equal(harness.calls.length, 0);
	} finally {
		harness.restore();
	}
});

test('analyticsStorageJustGranted is only true on the denied-to-granted edge', () => {
	assert.equal(analyticsStorageJustGranted(null, CONSENT_ANALYTICS), true);
	assert.equal(analyticsStorageJustGranted(CONSENT_DENIED, CONSENT_ANALYTICS), true);
	assert.equal(analyticsStorageJustGranted(CONSENT_ANALYTICS, CONSENT_ANALYTICS), false);
	assert.equal(analyticsStorageJustGranted(CONSENT_ANALYTICS, CONSENT_DENIED), false);
	assert.equal(analyticsStorageJustGranted(null, CONSENT_DENIED), false);
});

test('applyConsentState loads gtag only when analytics is newly granted', () => {
	const harness = installGtagHarness({ storage: memoryStorage() });
	try {
		applyConsentState({ ...CONSENT_DENIED });
		assert.deepEqual(harness.calls, [['consent', 'update', CONSENT_DENIED]]);
		assert.equal(harness.appended.length, 0);

		harness.calls.length = 0;
		applyConsentState({ ...CONSENT_ANALYTICS });
		assert.equal(harness.calls[0][0], 'consent');
		assert.equal(harness.calls[1][0], 'js');
		assert.deepEqual(harness.calls[2], ['config', 'G-ABC123']);
		assert.equal(harness.appended.length, 1);
		assert.equal(
			harness.appended[0].src,
			'https://www.googletagmanager.com/gtag/js?id=G-ABC123',
		);
		assert.equal(globalThis.__ciceroneGtagLoaded, true);

		harness.calls.length = 0;
		applyConsentState({ ...CONSENT_ANALYTICS });
		assert.deepEqual(harness.calls, [['consent', 'update', CONSENT_ANALYTICS]]);
	} finally {
		harness.restore();
	}
});

test('loadGoogleTag retries when script inject throws', () => {
	const harness = installGtagHarness({
		document: {
			createElement: () => ({ async: false, src: '' }),
			head: {
				appendChild() {
					throw new Error('blocked');
				},
			},
		},
	});
	try {
		loadGoogleTag();
		assert.equal(globalThis.__ciceroneGtagLoaded, false);
		assert.equal(harness.calls.length, 0);

		globalThis.document = {
			createElement: (tag) => ({ tagName: tag, async: false, src: '' }),
			head: {
				appendChild: (node) => {
					harness.appended.push(node);
					return node;
				},
			},
		};
		loadGoogleTag();
		assert.equal(globalThis.__ciceroneGtagLoaded, true);
		assert.equal(harness.calls[0][0], 'js');
		assert.deepEqual(harness.calls[1], ['config', 'G-ABC123']);
		assert.equal(harness.appended.length, 1);
	} finally {
		harness.restore();
	}
});

test('initConsentBanner loads gtag when stored analytics consent is granted', () => {
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
	const harness = installGtagHarness({
		storage: memoryStorage({
			[CONSENT_STORAGE_KEY]: JSON.stringify(CONSENT_ANALYTICS),
		}),
		document: doc,
	});
	try {
		initConsentBanner(doc);
		assert.equal(root.hidden, true);
		assert.equal(harness.calls[0][0], 'js');
		assert.deepEqual(harness.calls[1], ['config', 'G-ABC123']);
		assert.equal(globalThis.__ciceroneGtagLoaded, true);
	} finally {
		harness.restore();
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

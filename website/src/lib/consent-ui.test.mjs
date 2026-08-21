import assert from 'node:assert/strict';
import { test } from 'node:test';

import { CONSENT_DENIED } from './consent.mjs';
import {
	readStoredConsent,
	tabWrapTarget,
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

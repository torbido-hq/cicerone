import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
	CONSENT_ANALYTICS,
	CONSENT_DENIED,
	CONSENT_KEYS,
	CONSENT_STORAGE_KEY,
	analyticsHead,
	buildConsentInitScript,
	buildGtagConfigScript,
	canonicalGaMeasurementId,
	GA_MEASUREMENT_ID,
	gaMeasurementId,
	isGaMeasurementId,
	parseStoredConsent,
} from './consent.mjs';

test('canonicalGaMeasurementId trims and uppercases valid ids', () => {
	assert.equal(canonicalGaMeasurementId('G-ABC123'), 'G-ABC123');
	assert.equal(canonicalGaMeasurementId(' gt-xyz9 '), 'GT-XYZ9');
	assert.equal(canonicalGaMeasurementId(''), '');
	assert.equal(canonicalGaMeasurementId('UA-123'), '');
	assert.equal(canonicalGaMeasurementId('G-'), '');
});

test('isGaMeasurementId accepts G- and GT- ids', () => {
	assert.equal(isGaMeasurementId('G-ABC123'), true);
	assert.equal(isGaMeasurementId('GT-XYZ9'), true);
	assert.equal(isGaMeasurementId('g-abc123'), true);
	assert.equal(isGaMeasurementId('gt-xyz9'), true);
	assert.equal(isGaMeasurementId(' G-ABC123 '), true);
	assert.equal(isGaMeasurementId(''), false);
	assert.equal(isGaMeasurementId('UA-123'), false);
	assert.equal(isGaMeasurementId('G-'), false);
});

test('parseStoredConsent requires all consent-mode v2 keys', () => {
	assert.equal(parseStoredConsent(null), null);
	assert.equal(parseStoredConsent(''), null);
	assert.equal(parseStoredConsent('{'), null);
	assert.equal(parseStoredConsent('[]'), null);
	assert.equal(parseStoredConsent(JSON.stringify({ analytics_storage: 'granted' })), null);
	assert.deepEqual(parseStoredConsent(JSON.stringify(CONSENT_ANALYTICS)), {
		...CONSENT_ANALYTICS,
	});
	assert.deepEqual(parseStoredConsent(JSON.stringify(CONSENT_DENIED)), {
		...CONSENT_DENIED,
	});
	assert.deepEqual(Object.keys(CONSENT_DENIED), [...CONSENT_KEYS]);
});

test('gaMeasurementId defaults to the cicerone.dev tag', () => {
	assert.equal(gaMeasurementId(), GA_MEASUREMENT_ID);
	assert.equal(GA_MEASUREMENT_ID, 'G-E38EP8PJSR');
});

test('gaMeasurementId prefers a valid PUBLIC_GA_MEASUREMENT_ID from env', () => {
	assert.equal(gaMeasurementId({ PUBLIC_GA_MEASUREMENT_ID: 'G-OVERRIDE1' }), 'G-OVERRIDE1');
	assert.equal(gaMeasurementId({ PUBLIC_GA_MEASUREMENT_ID: 'g-override1' }), 'G-OVERRIDE1');
	assert.equal(gaMeasurementId({ PUBLIC_GA_MEASUREMENT_ID: 'nope' }), GA_MEASUREMENT_ID);
	assert.equal(gaMeasurementId({}), GA_MEASUREMENT_ID);
});

test('buildConsentInitScript sets denied defaults before any update', () => {
	const script = buildConsentInitScript('G-ABC123');
	assert.match(script, /consent','default'/);
	assert.match(script, /"analytics_storage":"denied"/);
	assert.match(script, /"ad_user_data":"denied"/);
	assert.match(script, /ads_data_redaction/);
	assert.match(script, /var CONSENT_KEYS=/);
	assert.match(script, new RegExp(CONSENT_STORAGE_KEY));
	assert.match(script, /try \{ raw=localStorage\.getItem/);
	assert.match(script, /catch \(e\) \{\}/);
	assert.match(script, /__CICERONE_GA_ID="G-ABC123"/);
	const defaultAt = script.indexOf("consent','default'");
	const updateAt = script.indexOf("consent','update'");
	assert.ok(defaultAt >= 0 && updateAt > defaultAt);
});

test('buildGtagConfigScript quotes the measurement id', () => {
	assert.equal(
		buildGtagConfigScript('G-ABC123'),
		`gtag('js',new Date());gtag('config',"G-ABC123");`,
	);
	assert.equal(
		buildGtagConfigScript(' g-abc123 '),
		`gtag('js',new Date());gtag('config',"G-ABC123");`,
	);
	assert.equal(buildGtagConfigScript(''), '');
	assert.equal(buildGtagConfigScript('UA-123'), '');
});

test('analyticsHead is empty without a measurement id', () => {
	assert.deepEqual(analyticsHead(''), []);
});

test('analyticsHead uses the default measurement id', () => {
	const head = analyticsHead();
	assert.equal(head.length, 1);
	assert.match(String(head[0].content), new RegExp(`__CICERONE_GA_ID="${GA_MEASUREMENT_ID}"`));
	assert.equal(head[0].attrs, undefined);
});

test('analyticsHead emits consent defaults without loading gtag.js', () => {
	const head = analyticsHead('G-ABC123');
	assert.equal(head.length, 1);
	assert.match(String(head[0].content), /consent','default'/);
	assert.match(String(head[0].content), /__CICERONE_GA_ID="G-ABC123"/);
	assert.doesNotMatch(JSON.stringify(head), /googletagmanager\.com\/gtag\/js/);
	assert.match(String(analyticsHead('g-abc123')[0].content), /__CICERONE_GA_ID="G-ABC123"/);
});

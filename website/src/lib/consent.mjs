export const CONSENT_STORAGE_KEY = 'cicerone-consent';

export const CONSENT_KEYS = Object.freeze([
	'ad_storage',
	'ad_user_data',
	'ad_personalization',
	'analytics_storage',
]);

export const CONSENT_DENIED = Object.freeze(
	Object.fromEntries(CONSENT_KEYS.map((key) => [key, 'denied'])),
);

export const CONSENT_ANALYTICS = Object.freeze({
	...CONSENT_DENIED,
	analytics_storage: 'granted',
});

export function canonicalGaMeasurementId(value) {
	const id = String(value ?? '').trim().toUpperCase();
	return /^(?:G|GT)-[A-Z0-9]+$/.test(id) ? id : '';
}

export function isGaMeasurementId(value) {
	return canonicalGaMeasurementId(value) !== '';
}

export const GA_MEASUREMENT_ID = 'G-E38EP8PJSR';

export function gaMeasurementId(env) {
	const sources = env
		? [env]
		: [
				(() => {
					try {
						return import.meta.env;
					} catch {
						return undefined;
					}
				})(),
				typeof process !== 'undefined' ? process.env : undefined,
			];
	for (const source of sources) {
		const id = canonicalGaMeasurementId(source?.PUBLIC_GA_MEASUREMENT_ID);
		if (id) return id;
	}
	return GA_MEASUREMENT_ID;
}

export function parseStoredConsent(raw) {
	if (raw == null || raw === '') return null;
	try {
		const parsed = JSON.parse(raw);
		if (parsed == null || typeof parsed !== 'object' || Array.isArray(parsed)) {
			return null;
		}
		/** @type {Record<string, string>} */
		const out = {};
		for (const key of CONSENT_KEYS) {
			const value = parsed[key];
			if (value !== 'granted' && value !== 'denied') return null;
			out[key] = value;
		}
		return out;
	} catch {
		return null;
	}
}

export function buildConsentInitScript(measurementId = gaMeasurementId()) {
	const id = canonicalGaMeasurementId(measurementId);
	return `(function(){
var CONSENT_KEYS=${JSON.stringify(CONSENT_KEYS)};
${parseStoredConsent.toString()}
try {
window.dataLayer=window.dataLayer||[];
window.gtag=window.gtag||function(){dataLayer.push(arguments);};
window.__CICERONE_GA_ID=${JSON.stringify(id)};
gtag('consent','default',${JSON.stringify({ ...CONSENT_DENIED, wait_for_update: 500 })});
gtag('set','ads_data_redaction',true);
var raw=null;
try { raw=localStorage.getItem(${JSON.stringify(CONSENT_STORAGE_KEY)}); } catch (e) {}
var stored=parseStoredConsent(raw);
if(stored)gtag('consent','update',stored);
} catch (e) {}
})();`;
}

export function buildGtagConfigScript(measurementId) {
	const id = canonicalGaMeasurementId(measurementId);
	if (!id) return '';
	return `gtag('js',new Date());gtag('config',${JSON.stringify(id)});`;
}

export function analyticsHead(measurementId = gaMeasurementId()) {
	const id = canonicalGaMeasurementId(measurementId);
	if (!id) return [];
	return [{ tag: 'script', content: buildConsentInitScript(id) }];
}

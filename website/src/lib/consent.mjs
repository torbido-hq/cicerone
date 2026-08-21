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

export function isGaMeasurementId(value) {
	return /^(?:G|GT)-[A-Z0-9]+$/.test(String(value || '').trim().toUpperCase());
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
		const value = String(source?.PUBLIC_GA_MEASUREMENT_ID ?? '').trim();
		if (isGaMeasurementId(value)) return value.toUpperCase();
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
	const id = String(measurementId || '').trim().toUpperCase();
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
	return `gtag('js',new Date());gtag('config',${JSON.stringify(String(measurementId || '').trim())});`;
}

export function analyticsHead(measurementId = gaMeasurementId()) {
	const id = String(measurementId || '').trim().toUpperCase();
	if (!isGaMeasurementId(id)) return [];
	return [{ tag: 'script', content: buildConsentInitScript(id) }];
}

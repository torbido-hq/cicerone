export const CONSENT_STORAGE_KEY = 'cicerone-consent';

export const CONSENT_DENIED = Object.freeze({
	ad_storage: 'denied',
	ad_user_data: 'denied',
	ad_personalization: 'denied',
	analytics_storage: 'denied',
});

export const CONSENT_ANALYTICS = Object.freeze({
	...CONSENT_DENIED,
	analytics_storage: 'granted',
});

export function isGaMeasurementId(value) {
	return /^(?:G|GT)-[A-Z0-9]+$/.test(String(value || '').trim());
}

export function gaMeasurementId() {
	try {
		return String(import.meta.env.PUBLIC_GA_MEASUREMENT_ID ?? '').trim();
	} catch {
		return '';
	}
}

export function parseStoredConsent(raw) {
	if (raw == null || raw === '') return null;
	try {
		const parsed = JSON.parse(raw);
		if (parsed == null || typeof parsed !== 'object' || Array.isArray(parsed)) {
			return null;
		}
		const keys = [
			'ad_storage',
			'ad_user_data',
			'ad_personalization',
			'analytics_storage',
		];
		/** @type {Record<string, string>} */
		const out = {};
		for (const key of keys) {
			const value = parsed[key];
			if (value !== 'granted' && value !== 'denied') return null;
			out[key] = value;
		}
		return out;
	} catch {
		return null;
	}
}

export function buildConsentInitScript() {
	return `(function(){
${parseStoredConsent.toString()}
try {
window.dataLayer=window.dataLayer||[];
window.gtag=window.gtag||function(){dataLayer.push(arguments);};
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
	if (!isGaMeasurementId(measurementId)) return [];
	return [
		{ tag: 'script', content: buildConsentInitScript() },
		{
			tag: 'script',
			attrs: {
				async: true,
				src: `https://www.googletagmanager.com/gtag/js?id=${measurementId}`,
			},
		},
		{ tag: 'script', content: buildGtagConfigScript(measurementId) },
	];
}

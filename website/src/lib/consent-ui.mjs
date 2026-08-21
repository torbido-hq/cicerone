import {
	CONSENT_ANALYTICS,
	CONSENT_DENIED,
	CONSENT_STORAGE_KEY,
	parseStoredConsent,
} from './consent.mjs';

const DIALOG_FOCUSABLE = 'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function readStoredConsent() {
	try {
		return parseStoredConsent(localStorage.getItem(CONSENT_STORAGE_KEY));
	} catch {
		return null;
	}
}

export function writeStoredConsent(state) {
	try {
		localStorage.setItem(CONSENT_STORAGE_KEY, JSON.stringify(state));
	} catch {
		// private mode, disabled storage, or non-browser
	}
}

export function updateGtagConsent(state) {
	try {
		const gtag = globalThis.gtag;
		if (typeof gtag === 'function') gtag('consent', 'update', state);
	} catch {
		// gtag or window blocked
	}
}

export function dialogFocusables(root) {
	return [...root.querySelectorAll(DIALOG_FOCUSABLE)];
}

export function tabWrapTarget(items, current, shiftKey) {
	if (!items.length) return null;
	const first = items[0];
	const last = items[items.length - 1];
	if (shiftKey && current === first) return last;
	if (!shiftKey && current === last) return first;
	return null;
}

export function trapDialogTab(event, root) {
	if (event.key !== 'Tab') return;
	const items = dialogFocusables(root);
	const wrap = tabWrapTarget(items, event.target, event.shiftKey);
	if (!wrap) return;
	event.preventDefault();
	wrap.focus();
}

export function openConsentDialog(root) {
	root.hidden = false;
	root.setAttribute('aria-modal', 'true');
	const first = dialogFocusables(root)[0];
	if (first && typeof first.focus === 'function') first.focus();
	else if (typeof root.focus === 'function') root.focus();
}

export function closeConsentDialog(root, previousFocus) {
	root.hidden = true;
	root.setAttribute('aria-modal', 'false');
	try {
		if (previousFocus && typeof previousFocus.focus === 'function') previousFocus.focus();
	} catch {
		// previous node unmounted
	}
}

function isElement(value) {
	return typeof Element !== 'undefined' && value instanceof Element;
}

export function initConsentBanner(doc = document) {
	const root = doc.getElementById('cicerone-consent');
	if (!root) return;

	const footer = doc.querySelector('[data-cicerone-consent-footer]');
	if (footer) footer.hidden = false;

	/** @type {Element | null} */
	let lastFocus = null;

	const apply = (state) => {
		writeStoredConsent(state);
		updateGtagConsent(state);
		closeConsentDialog(root, lastFocus);
		lastFocus = null;
	};

	const open = () => {
		lastFocus = isElement(doc.activeElement) ? doc.activeElement : null;
		openConsentDialog(root);
	};

	root.addEventListener('keydown', (event) => {
		if (root.hidden) return;
		trapDialogTab(event, root);
		if (event.key === 'Escape' && readStoredConsent()) {
			event.preventDefault();
			closeConsentDialog(root, lastFocus);
			lastFocus = null;
		}
	});

	root.querySelectorAll('[data-cicerone-consent]').forEach((button) => {
		button.addEventListener('click', () => {
			const granted = button.getAttribute('data-cicerone-consent') === 'granted';
			apply(granted ? { ...CONSENT_ANALYTICS } : { ...CONSENT_DENIED });
		});
	});

	doc.addEventListener('click', (event) => {
		const target = event.target;
		if (!isElement(target) || !target.closest('[data-cicerone-consent-open]')) {
			return;
		}
		event.preventDefault();
		open();
	});

	if (!readStoredConsent()) open();
}

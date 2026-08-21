/// <reference types="astro/client" />

interface Window {
	gtag?: (...args: unknown[]) => void;
	dataLayer?: unknown[];
	__CICERONE_GA_ID?: string;
	__ciceroneGtagLoaded?: boolean;
}

interface ImportMetaEnv {
	readonly PUBLIC_GA_MEASUREMENT_ID?: string;
}

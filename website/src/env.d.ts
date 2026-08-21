/// <reference types="astro/client" />

interface Window {
	gtag?: (...args: unknown[]) => void;
	dataLayer?: unknown[];
}

interface ImportMetaEnv {
	readonly PUBLIC_GA_MEASUREMENT_ID?: string;
}

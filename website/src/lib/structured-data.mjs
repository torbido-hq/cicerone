export const SITE_URL = 'https://cicerone.dev';
export const SITE_NAME = 'Cicerone';
export const SITE_DESCRIPTION =
	'Self-hosted batch recommender — hybrid models, policies, incremental events, serve API, and dashboard. No live inference in the request path.';
export const DEFAULT_OG_IMAGE = `${SITE_URL}/images/docs/dashboard.png`;
export const DEFAULT_OG_IMAGE_ALT = 'Cicerone recommendation job status dashboard';
export const DEFAULT_OG_IMAGE_WIDTH = 1920;
export const DEFAULT_OG_IMAGE_HEIGHT = 1080;

export const ARTICLES_DESCRIPTION =
	'Notes on running a batch recommender next to an existing shop: nightly tables, webhooks, and serve lookups.';
export const AUTHOR_DESCRIPTION = 'Articles by Nicholas Wieland about Cicerone, a self-hosted batch recommender.';

export const PUBLISHER = {
	'@type': 'Organization',
	name: SITE_NAME,
	url: SITE_URL,
	logo: {
		'@type': 'ImageObject',
		url: `${SITE_URL}/images/docs/cicerone-logo.svg`,
	},
};

export function homeStructuredData() {
	return {
		'@context': 'https://schema.org',
		'@graph': [
			{
				'@type': 'WebSite',
				'@id': `${SITE_URL}/#website`,
				name: SITE_NAME,
				url: `${SITE_URL}/`,
				description: SITE_DESCRIPTION,
				inLanguage: 'en',
				publisher: { '@id': `${SITE_URL}/#org` },
			},
			{
				'@type': 'SoftwareApplication',
				name: SITE_NAME,
				applicationCategory: 'DeveloperApplication',
				operatingSystem: 'Linux',
				url: SITE_URL,
				description: SITE_DESCRIPTION,
				offers: { '@type': 'Offer', price: '0', priceCurrency: 'USD' },
				publisher: { '@id': `${SITE_URL}/#org` },
			},
			{
				'@type': 'Organization',
				'@id': `${SITE_URL}/#org`,
				name: SITE_NAME,
				url: SITE_URL,
			},
		],
	};
}

export function articleImageForId(id) {
	if (id === 'articles/this-afternoons-checkout-can-move-the-row') {
		return `${SITE_URL}/images/afternoon-checkout-architecture.png`;
	}
	return DEFAULT_OG_IMAGE;
}

export function listingDescriptionForId(id) {
	if (id === 'articles') return ARTICLES_DESCRIPTION;
	if (typeof id === 'string' && id.startsWith('articles/authors/')) return AUTHOR_DESCRIPTION;
	return undefined;
}

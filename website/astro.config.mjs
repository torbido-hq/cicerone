// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	site: 'https://cicerone.dev',
	integrations: [
		starlight({
			title: 'Cicerone',
			description:
				'Self-hosted batch recommender (rectools + LightFM) with serve API, policies, AutoML, and dashboard.',
			favicon: '/favicon.svg',
			logo: {
				src: './src/assets/cicerone-logo.svg',
				alt: 'Cicerone',
				replacesTitle: true,
			},
			social: [
				{
					icon: 'github',
					label: 'GitHub',
					href: 'https://github.com/torbido-hq/cicerone',
				},
			],
			editLink: {
				baseUrl: 'https://github.com/torbido-hq/cicerone/edit/main/',
			},
			customCss: ['./src/styles/custom.css'],
			sidebar: [
				{ label: 'Home', link: '/' },
				{
					label: 'Guides',
					items: [
						{ label: 'Tutorial', slug: 'tutorial' },
						{ label: 'Architecture', slug: 'architecture' },
					],
				},
				{
					label: 'Reference',
					items: [
						{
							label: 'Serve OpenAPI',
							link: '/openapi/serve.openapi.json',
							attrs: { target: '_blank', rel: 'noopener noreferrer' },
						},
						{
							label: 'Changelog',
							link: 'https://github.com/torbido-hq/cicerone/blob/main/CHANGELOG.md',
							attrs: { target: '_blank', rel: 'noopener noreferrer' },
						},
						{
							label: 'License',
							link: 'https://github.com/torbido-hq/cicerone/blob/main/LICENSE',
							attrs: { target: '_blank', rel: 'noopener noreferrer' },
						},
						{
							label: 'Repository README',
							link: 'https://github.com/torbido-hq/cicerone/blob/main/README.md',
							attrs: { target: '_blank', rel: 'noopener noreferrer' },
						},
					],
				},
			],
			head: [
				{
					tag: 'meta',
					attrs: {
						property: 'og:image',
						content: 'https://cicerone.dev/images/dashboard.png',
					},
				},
				{
					tag: 'meta',
					attrs: {
						property: 'og:image:alt',
						content: 'Cicerone recommendation job status dashboard',
					},
				},
				{
					tag: 'meta',
					attrs: { name: 'twitter:card', content: 'summary_large_image' },
				},
				{
					tag: 'meta',
					attrs: {
						name: 'twitter:image',
						content: 'https://cicerone.dev/images/dashboard.png',
					},
				},
				{
					tag: 'link',
					attrs: { rel: 'sitemap', href: '/sitemap-index.xml' },
				},
			],
		}),
	],
});

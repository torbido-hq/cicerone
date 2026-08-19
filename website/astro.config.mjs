// @ts-check
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightBlog from 'starlight-blog';
import {
	ARTICLES_PREFIX,
	articlesContentDir,
	hasPublishedArticles,
} from './src/lib/articles.mjs';

const websiteRoot = dirname(fileURLToPath(import.meta.url));
const articlesDir = articlesContentDir(websiteRoot);

const articlesPlugin = hasPublishedArticles(articlesDir)
	? [
			starlightBlog({
				title: 'Articles',
				prefix: ARTICLES_PREFIX,
				metrics: { readingTime: true },
				authors: {
					nicholas: {
						name: 'Nicholas Wieland',
						url: 'https://github.com/ngw',
					},
				},
			}),
		]
	: [];

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
				light: './src/assets/cicerone-logo.svg',
				dark: './src/assets/cicerone-logo-dark.svg',
				alt: 'Cicerone',
				replacesTitle: true,
			},
			plugins: articlesPlugin,
			social: [
				{
					icon: 'github',
					label: 'GitHub',
					href: 'https://github.com/torbido-hq/cicerone',
				},
			],
			components: {
				PageFrame: './src/components/PageFrame.astro',
			},
			customCss: ['./src/styles/custom.css'],
			sidebar: [
				{ label: 'Home', link: '/' },
				{
					label: 'Guides',
					items: [
						{ label: 'Tutorial', slug: 'tutorial' },
						{ label: 'Architecture', slug: 'architecture' },
						{ label: 'Incremental events', slug: 'incremental-events' },
					],
				},
				{
					label: 'Reference',
					items: [
						{
							label: 'Serve OpenAPI',
							link: '/openapi/',
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
					tag: 'link',
					attrs: { rel: 'preconnect', href: 'https://fonts.googleapis.com' },
				},
				{
					tag: 'link',
					attrs: {
						rel: 'preconnect',
						href: 'https://fonts.gstatic.com',
						crossorigin: 'anonymous',
					},
				},
				{
					tag: 'link',
					attrs: {
						rel: 'stylesheet',
						href: 'https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:ital,wght@0,400;0,500;0,600;1,400&display=swap',
					},
				},
				{
					tag: 'meta',
					attrs: {
						property: 'og:image',
						content: 'https://cicerone.dev/images/docs/dashboard.png',
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
						content: 'https://cicerone.dev/images/docs/dashboard.png',
					},
				},
				{
					tag: 'meta',
					attrs: {
						name: 'twitter:image:alt',
						content: 'Cicerone recommendation job status dashboard',
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

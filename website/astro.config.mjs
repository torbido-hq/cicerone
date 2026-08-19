// @ts-check
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightBlog from 'starlight-blog';

const websiteRoot = dirname(fileURLToPath(import.meta.url));
const articlesDir = join(websiteRoot, 'src/content/docs/articles');

function hasPublishedArticles() {
	if (!existsSync(articlesDir)) return false;
	for (const name of readdirSync(articlesDir)) {
		if (name.startsWith('_') || !/\.(md|mdx)$/i.test(name)) continue;
		const text = readFileSync(join(articlesDir, name), 'utf8');
		const fm = /^---\r?\n([\s\S]*?)\r?\n---/.exec(text);
		if (
			fm &&
			/^draft:\s*true\s*$/m.test(fm[1]) &&
			process.env.NODE_ENV === 'production'
		) {
			continue;
		}
		return true;
	}
	return false;
}

const articlesPlugin = hasPublishedArticles()
	? [
			starlightBlog({
				title: 'Articles',
				prefix: 'articles',
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
					tag: 'link',
					attrs: { rel: 'sitemap', href: '/sitemap-index.xml' },
				},
			],
		}),
	],
});

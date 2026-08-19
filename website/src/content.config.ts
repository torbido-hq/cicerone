import { defineCollection } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';
import { blogSchema } from 'starlight-blog/schema';

export const collections = {
	// starlight-blog posts are Starlight docs pages (`articles/`); schema is partial.
	docs: defineCollection({
		loader: docsLoader(),
		schema: docsSchema({ extend: (context) => blogSchema(context) }),
	}),
};

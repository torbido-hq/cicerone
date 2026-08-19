import { defineCollection } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';
import { blogSchema } from 'starlight-blog/schema';

export const collections = {
	// starlight-blog only reads the Starlight `docs` collection (`articles/` prefix).
	// A second collection would be invisible to the plugin. blogSchema is .partial().
	docs: defineCollection({
		loader: docsLoader(),
		schema: docsSchema({ extend: (context) => blogSchema(context) }),
	}),
};

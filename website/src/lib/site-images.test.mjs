import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';

const imagesDir = join(dirname(fileURLToPath(import.meta.url)), '../../public/images');

test('hand-owned public SVGs are UTF-8 so <img> XML parse succeeds', () => {
	const svgs = readdirSync(imagesDir).filter((name) => name.endsWith('.svg'));
	assert.ok(svgs.includes('flow.svg'));
	const decoder = new TextDecoder('utf-8', { fatal: true });
	for (const name of svgs) {
		const text = decoder.decode(readFileSync(join(imagesDir, name)));
		assert.match(text, /<svg\b/, `${name} is an SVG`);
	}
});

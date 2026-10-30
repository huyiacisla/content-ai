// usage: JEV_AI_API_KEY=... node src/cli.mjs request.json   (request.json = { state, questions, model? })
import { readFileSync } from 'node:fs';
import { createClient } from './client.mjs';

const file = process.argv[2];
if (!file) { console.error('usage: node src/cli.mjs <request.json>'); process.exit(2); }
const res = await createClient().decide(JSON.parse(readFileSync(file, 'utf8')));
console.log(JSON.stringify(res, null, 2));

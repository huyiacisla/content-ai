import test from 'node:test';
import assert from 'node:assert/strict';
import { createClient, JevError, validate } from '../src/client.mjs';

const ok = (body, status = 200, headers = {}) => async (url, init) => {
  ok.last = { url, init };
  return new Response(JSON.stringify(body), { status, headers });
};
const q = { hook: { type: 'noul', instructions: 'Is the hook strong?' } };

test('decide posts to /v1/systemone with bearer auth', async () => {
  const f = ok({ model: 'jev-1.14.0', answers: { hook: 0.8 }, usage: { input_tokens: 5, output_tokens: 1 } });
  const c = createClient({ apiKey: 'k', fetch: f });
  const r = await c.decide({ state: 'x', questions: q });
  assert.equal(r.answers.hook, 0.8);
  assert.equal(ok.last.url, 'https://jev-ai.pro/api/v1/systemone');
  assert.equal(ok.last.init.headers.Authorization, 'Bearer k');
  assert.equal(JSON.parse(ok.last.init.body).model, 'jev-latest');
});

test('non-2xx throws JevError with retryAfter', async () => {
  const c = createClient({ apiKey: 'k', fetch: ok({ error: 'slow' }, 429, { 'retry-after': '3' }) });
  await assert.rejects(c.decide({ state: 'x', questions: q }), (e) => e instanceof JevError && e.status === 429 && e.retryAfter === '3');
});

test('validate rejects bad input', () => {
  assert.throws(() => validate({ questions: q }), /state/);
  assert.throws(() => validate({ state: 1, questions: {} }), /1\.\.64/);
  assert.throws(() => validate({ state: 1, questions: { a: { type: 'x', instructions: 'i' } } }), /type/);
});

test('missing api key throws', () => {
  const prev = process.env.JEV_AI_API_KEY; delete process.env.JEV_AI_API_KEY;
  assert.throws(() => createClient(), /JEV_AI_API_KEY/);
  if (prev) process.env.JEV_AI_API_KEY = prev;
});

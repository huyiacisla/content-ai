// Jev AI client — https://jev-ai.pro/docs#endpoint
// Auth: JEV_AI_API_KEY (server-side only). No auto-retry: a timed-out request may still have been processed.

export const BASE_URL = 'https://jev-ai.pro/api';
export const QUESTION_TYPES = ['noul', 'choice', 'score'];
const MAX_QUESTIONS = 64;
const MAX_BODY_BYTES = 256_000;

export class JevError extends Error {
  constructor(status, body, retryAfter) {
    super(`Jev AI ${status}: ${typeof body === 'string' ? body : JSON.stringify(body)}`);
    this.status = status;
    this.body = body;
    this.retryAfter = retryAfter;
  }
}

export function validate({ state, questions }) {
  if (state === undefined || state === null) throw new Error('state is required');
  const ids = Object.keys(questions ?? {});
  if (ids.length < 1 || ids.length > MAX_QUESTIONS) throw new Error(`questions must hold 1..${MAX_QUESTIONS} entries`);
  for (const id of ids) {
    const q = questions[id];
    if (!QUESTION_TYPES.includes(q?.type)) throw new Error(`question "${id}": type must be one of ${QUESTION_TYPES.join('|')}`);
    if (!q.instructions) throw new Error(`question "${id}": instructions is required`);
  }
}

export function createClient({ apiKey = process.env.JEV_AI_API_KEY, baseURL = BASE_URL, model = 'jev-latest', fetch: f = fetch } = {}) {
  if (!apiKey) throw new Error('JEV_AI_API_KEY is not set');

  async function request(path, { method = 'POST', body } = {}) {
    const payload = body === undefined ? undefined : JSON.stringify(body);
    if (payload && Buffer.byteLength(payload) > MAX_BODY_BYTES) throw new Error(`request body exceeds ${MAX_BODY_BYTES} bytes`);
    const res = await f(`${baseURL}${path}`, {
      method,
      headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
      body: payload,
    });
    const text = await res.text();
    let data;
    try { data = text ? JSON.parse(text) : null; } catch { data = text; }
    if (!res.ok) throw new JevError(res.status, data, res.headers.get('retry-after'));
    return data;
  }

  return {
    // -> { model, answers: { [id]: ... }, usage: { input_tokens, output_tokens } }
    decide: ({ state, questions, model: m = model }) => {
      validate({ state, questions });
      return request('/v1/systemone', { body: { model: m, state, questions } });
    },
    // saved judge: judgeId (+ optional revision) instead of questions
    decideWithJudge: ({ state, judgeId, revision, model: m = model }) =>
      request('/v1/systemone', { body: { model: m, state, judgeId, ...(revision !== undefined && { revision }) } }),
    models: () => request('/v1/models', { method: 'GET' }),
  };
}

# jev-ai

Client Jev AI (`@typesafe-ai/sdk` 0.6.0, baseURL `https://jev-ai.pro/api`, retry tắt).

```bash
cd jev-ai && npm install
JEV_AI_API_KEY=... npm run check   # GET /v1/models, không inference
```

```js
import { createJevClient, noul } from "./client.mjs";
const jev = createJevClient();
const r = await jev.systemOne({ state: "...", questions: { ok: noul("...?") } });
```

## Client không phụ thuộc SDK
`src/client.mjs` (fetch thuần) + `src/cli.mjs`; test offline: `npm test`.
`npm run decide -- request.json` gọi `POST /v1/systemone`.

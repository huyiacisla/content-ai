import { TypeSafeClient } from "@typesafe-ai/sdk";

export const JEV_BASE_URL = "https://jev-ai.pro/api";

// Key đọc từ JEV_AI_API_KEY; baseURL phải trỏ jev-ai.pro (SDK tự nối /v1/systemone).
// retry tắt: decision request không retry tự động.
export function createJevClient({ apiKey = process.env.JEV_AI_API_KEY, ...rest } = {}) {
  if (!apiKey) throw new Error("Thiếu JEV_AI_API_KEY");
  return new TypeSafeClient({
    apiKey,
    baseURL: process.env.JEV_AI_BASE_URL ?? JEV_BASE_URL,
    retry: { maxRetries: 0 },
    ...rest,
  });
}

export { choice, noul, score } from "@typesafe-ai/sdk";

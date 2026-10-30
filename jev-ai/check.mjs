import { createJevClient } from "./client.mjs";

const client = createJevClient();
console.log("baseURL:", client.baseURL);
const models = await client.models.list();
console.log(models.map((m) => m.name ?? m.id ?? m));

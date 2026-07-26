import { Container } from "@cloudflare/containers";

interface Env {
  GRADIO_CONTAINER: DurableObjectNamespace<GradioContainer>;
  FRAMES: R2Bucket;
  ANTHROPIC_API_KEY: string;
}

// Public URL of this Worker — used by the container to build absolute R2
// image URLs (gallery) and to fetch candidate frames in Stage 2 vision.
const PUBLIC_BASE_URL = "https://zhenhuan.aidigest.workers.dev";

export class GradioContainer extends Container {
  defaultPort = 8080;
  sleepAfter = "30m";
  enableInternet = true; // Anthropic API + R2 frame fetch in Stage 2

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    // Container env vars must be set on the class (the wrangler.jsonc
    // `envVars` field is not honored). ANTHROPIC_API_KEY comes from a
    // Worker secret (`wrangler secret put`), never committed.
    this.envVars = {
      PUBLIC_BASE_URL,
      ANTHROPIC_API_KEY: env.ANTHROPIC_API_KEY,
    };
  }
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    // /frames/* -> R2, with edge caching
    if (url.pathname.startsWith("/frames/")) {
      const key = url.pathname.slice("/frames/".length);
      if (!key) return new Response("Not found", { status: 404 });

      const cache = caches.default;
      const cached = await cache.match(request);
      if (cached) return cached;

      const object = await env.FRAMES.get(key);
      if (object === null) return new Response("Not found", { status: 404 });

      const headers = new Headers();
      object.writeHttpMetadata(headers);
      headers.set("etag", object.httpEtag);
      headers.set("Content-Type", "image/jpeg");
      headers.set("Cache-Control", "public, max-age=31536000, immutable");

      const response = new Response(object.body, { headers });
      ctx.waitUntil(cache.put(request, response.clone()));
      return response;
    }

    // everything else -> the Gradio container (single shared instance)
    const container = env.GRADIO_CONTAINER.getByName("singleton");
    await container.startAndWaitForPorts();
    return container.fetch(request); // fetch(), NOT containerFetch() — SSE/WS
  },
};

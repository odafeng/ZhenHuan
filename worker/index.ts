import { Container } from "@cloudflare/containers";

interface Env {
  GRADIO_CONTAINER: DurableObjectNamespace<GradioContainer>;
  RATE_LIMITER: DurableObjectNamespace<RateLimiter>;
  FRAMES: R2Bucket;
  ANTHROPIC_API_KEY: string;
}

// Public URL of this Worker — used by the container to build absolute R2
// image URLs (gallery) and to fetch candidate frames in Stage 2 vision.
const PUBLIC_BASE_URL = "https://zhenhuan.aidigest.workers.dev";

// Per-IP search cap: N searches per WINDOW seconds.
const SEARCH_LIMIT = 8;
const SEARCH_WINDOW_MS = 60_000;

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

// Fixed-window rate limiter, one instance per IP. Durable Objects give a
// single-threaded, consistent counter — unlike the Workers rate-limit
// binding, which did not enforce reliably here.
export class RateLimiter {
  constructor(private state: DurableObjectState) {}

  async fetch(): Promise<Response> {
    const now = Date.now();
    const rec =
      (await this.state.storage.get<{ count: number; reset: number }>("w")) ??
      { count: 0, reset: now + SEARCH_WINDOW_MS };
    if (now >= rec.reset) {
      rec.count = 0;
      rec.reset = now + SEARCH_WINDOW_MS;
    }
    rec.count++;
    await this.state.storage.put("w", rec);
    return Response.json({
      allowed: rec.count <= SEARCH_LIMIT,
      retryAfter: Math.max(1, Math.ceil((rec.reset - now) / 1000)),
    });
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

    // Per-IP cap on the search REST endpoint (POST /gradio_api/call/*), which
    // runs the paid reranker. This stops casual curl-the-REST-API abuse before
    // it reaches the container or Anthropic.
    //
    // KNOWN GAP: the browser AND gradio_client trigger search via
    // /gradio_api/queue/join, NOT /call/, so a client using the queue protocol
    // bypasses this cap. We do not limit /queue/join because the gr.Timer
    // readiness poll (~30/min) also flows through it, and telling them apart
    // needs body parsing. The hard cost ceiling is the Anthropic monthly spend
    // cap; move this limit inside the container's search() for full coverage.
    if (request.method === "POST" && url.pathname.startsWith("/gradio_api/call/")) {
      const ip = request.headers.get("CF-Connecting-IP") ?? "unknown";
      const stub = env.RATE_LIMITER.get(env.RATE_LIMITER.idFromName(ip));
      const { allowed, retryAfter } = (await stub.fetch("https://rl/").then((r) => r.json())) as {
        allowed: boolean;
        retryAfter: number;
      };
      if (!allowed) {
        return new Response("Rate limit exceeded — please wait a moment before searching again.", {
          status: 429,
          headers: { "Retry-After": String(retryAfter) },
        });
      }
    }

    // everything else -> the Gradio container (single shared instance)
    const container = env.GRADIO_CONTAINER.getByName("singleton");
    await container.startAndWaitForPorts();
    return container.fetch(request); // fetch(), NOT containerFetch() — SSE/WS
  },
};

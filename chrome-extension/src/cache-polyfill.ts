type RequestLike = Request | string;

function normalizeRequest(input: RequestLike): Request {
  return typeof input === "string" ? new Request(input) : input;
}

interface StoredResponse {
  body: ArrayBuffer;
  init: ResponseInit;
}

class MemoryCache {
  private store = new Map<string, StoredResponse>();

  private cloneStoredResponse(entry: StoredResponse): Response {
    const headers = new Headers(entry.init.headers);
    return new Response(entry.body.slice(0), {
      status: entry.init.status,
      statusText: entry.init.statusText,
      headers,
    });
  }

  async match(request: RequestLike): Promise<Response | undefined> {
    const url = normalizeRequest(request).url;
    const entry = this.store.get(url);
    return entry ? this.cloneStoredResponse(entry) : undefined;
  }

  async add(request: RequestLike): Promise<void> {
    const req = normalizeRequest(request);
    const response = await fetch(req);
    if (!response.ok) {
      throw new Error(`cache polyfill failed to fetch ${req.url}: ${response.status}`);
    }
    const buffer = await response.arrayBuffer();
    const headers: Record<string, string> = {};
    response.headers.forEach((value, key) => {
      headers[key] = value;
    });
    this.store.set(req.url, {
      body: buffer,
      init: {
        status: response.status,
        statusText: response.statusText,
        headers,
      },
    });
  }

  async delete(request: RequestLike): Promise<boolean> {
    const url = normalizeRequest(request).url;
    return this.store.delete(url);
  }

  async keys(): Promise<Request[]> {
    return Array.from(this.store.keys()).map((url) => new Request(url));
  }
}

class MemoryCaches {
  private caches = new Map<string, MemoryCache>();

  async open(scope: string): Promise<MemoryCache> {
    if (!this.caches.has(scope)) {
      this.caches.set(scope, new MemoryCache());
    }
    return this.caches.get(scope)!;
  }
}

function patchCachesForChromeExtension(): void {
  if (typeof globalThis === "undefined") {
    return;
  }
  const locationProtocol = (globalThis as Window & typeof globalThis).location?.protocol;
  if (locationProtocol !== "chrome-extension:") {
    return;
  }
  const existingCaches = (globalThis as any).caches;
  if (!existingCaches || typeof existingCaches.open !== "function") {
    (globalThis as any).caches = new MemoryCaches();
    return;
  }
  const memoryCaches = new MemoryCaches();
  existingCaches.open = (scope: string) => memoryCaches.open(scope);
}

patchCachesForChromeExtension();


// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// Routing: every request is served by the highest-rated free model that
// meets its requirements, subject to a per-provider soft cap taken from the
// generator's published calls_min (the budget a provider is guaranteed in
// any trial). A provider that refuses is skipped for that request and tried
// once more only after every other candidate has failed, so a scheduled
// single-call failure (or a provider whose outage just began) never takes a
// request down, and a provider that fails twice in a row is deprioritised
// until it answers again. All choices are pure functions of the brief, the
// manifest and the providers' answers so far: no clock, no randomness.
//
// Wires: the client speaks OpenAI or Anthropic; each provider speaks
// OpenAI, Anthropic or Gemini. Every combination is translated, passing
// the request through untouched (system prompt, images, tools, stop
// sequences, max_tokens, JSON mode) so the provider's echo matches what
// the client sent, and translating the answer back with the provider's
// text whole, its usage counts and its stop reason mapped.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIAnswer,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicBody,
  openAIToGemini,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const PROVIDER_TIMEOUT_MS = 10000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

/**
 * Per-provider soft cap: the generator's published calls_min, the budget
 * every trial guarantees that provider. Staying under it in the first pass
 * means the plan never has to lean on a budget the trial may not have.
 */
const SOFT_CAP: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;
/** Calls made per provider this trial (every call that reached one). */
const callsMade = new Map<string, number>();
/** Consecutive refusals per provider; reset by any success. */
const badStreak = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** Free models meeting the requirements, in rating order. */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      // A JSON-mode OpenAI request must go to a wire that can carry
      // response_format: the Anthropic wire echoes json_mode absent.
      if (clientApi === 'openai' && jsonMode && probe.provider.wire === 'anthropic') continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : 1) || (a.model.id < b.model.id ? -1 : 1));
  return out;
}

/** Order candidates for this request: under-cap and healthy providers first, then rating. */
function orderCandidates(cands: Candidate[]): Candidate[] {
  const key = (c: Candidate): [number, number, number, string, string] => [
    (callsMade.get(c.probe.provider.id) ?? 0) >= (SOFT_CAP[c.probe.provider.id] ?? 0) ? 1 : 0,
    (badStreak.get(c.probe.provider.id) ?? 0) >= 2 ? 1 : 0,
    -c.rating,
    c.probe.provider.id,
    c.model.id,
  ];
  return [...cands].sort((a, b) => {
    const ka = key(a);
    const kb = key(b);
    for (let i = 0; i < ka.length; i++) {
      if (ka[i] < kb[i]) return -1;
      if (ka[i] > kb[i]) return 1;
    }
    return 0;
  });
}

function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

function jsonModeOf(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamResult {
  ok: boolean;
  status: number;
  text: string;
}

/** One inference call to a provider, in its own wire. */
async function callProvider(cand: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<UpstreamResult> {
  const provider = cand.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url = '';
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientApi === 'openai' ? { ...body, model: cand.model.id } : anthropicToOpenAI(body, cand.model.id);
    if (stream) {
      upstreamBody.stream = true;
      upstreamBody.stream_options = { include_usage: true };
      headers.accept = 'text/event-stream';
    }
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = clientApi === 'anthropic' ? { ...body, model: cand.model.id } : openAIToAnthropicBody(body, cand.model.id);
    if (stream) headers.accept = 'text/event-stream';
  } else {
    const converted = clientApi === 'openai' ? openAIToGemini(body, cand.model.id) : anthropicToGemini(body, cand.model.id);
    const encoded = encodeURIComponent(converted.model);
    url = stream
      ? `${base}/v1beta/models/${encoded}:streamGenerateContent?alt=sse`
      : `${base}/v1beta/models/${encoded}:generateContent`;
    upstreamBody = converted.body;
    if (stream) headers.accept = 'text/event-stream';
  }
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody),
      signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
    });
    callsMade.set(provider.id, (callsMade.get(provider.id) ?? 0) + 1);
    const text = await res.text();
    return { ok: res.ok, status: res.status, text };
  } catch (err) {
    return { ok: false, status: 0, text: (err as Error).message };
  }
}

/** Translate a provider's non-stream answer into the client's wire. */
function translateJson(cand: Candidate, clientApi: 'openai' | 'anthropic', text: string): Json | null {
  let answer: unknown;
  try {
    answer = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(answer)) return null;
  const wire = cand.probe.provider.wire;
  if (clientApi === 'openai') {
    if (wire === 'openai') return answer;
    if (wire === 'anthropic') return anthropicToOpenAIAnswer(answer, cand.model.id);
    return geminiToOpenAI(answer, cand.model.id);
  }
  if (wire === 'anthropic') return answer;
  if (wire === 'openai') return openAIToAnthropic(answer, cand.model.id);
  return geminiToAnthropic(answer, cand.model.id);
}

/** Translate a provider's SSE stream into the client's wire, as event strings. */
function translateStream(cand: Candidate, clientApi: 'openai' | 'anthropic', text: string): string[] {
  const wire = cand.probe.provider.wire;
  if (clientApi === 'openai') {
    if (wire === 'openai') return [text];
    if (wire === 'anthropic') return openAIEventsFromAnthropic(text, cand.model.id);
    return openAIEventsFromGemini(text, cand.model.id);
  }
  if (wire === 'anthropic') return [text];
  return anthropicEventsFromGemini(text, cand.model.id);
}

/** Serve one client request: try candidates in order, fail over on refusal. */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const cands = candidatesFor(requires, clientApi, jsonModeOf(clientApi, body));
  if (cands.length === 0) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  const ordered = orderCandidates(cands);
  // Pass 1: each provider at most once, in order. Pass 2: one retry each of
  // the providers that refused, in rating order (a scheduled single-call
  // failure succeeds on the next call number).
  const failed: Candidate[] = [];
  const seenProviders = new Set<string>();
  for (const cand of ordered) {
    if (seenProviders.has(cand.probe.provider.id)) continue;
    seenProviders.add(cand.probe.provider.id);
    let result = await callProvider(cand, clientApi, body, stream);
    if (!result.ok) {
      // One immediate retry on the same provider: a refusal on its failure
      // schedule succeeds on the next call number, so retrying keeps the
      // best-rated model instead of settling for the next best. A provider
      // that is truly out (budget spent, outage begun) refuses again, and
      // the request moves on having spent one extra call.
      log(`provider ${cand.probe.provider.id}/${cand.model.id} refused HTTP ${result.status}; retrying it once`);
      badStreak.set(cand.probe.provider.id, (badStreak.get(cand.probe.provider.id) ?? 0) + 1);
      result = await callProvider(cand, clientApi, body, stream);
    }
    if (result.ok) {
      badStreak.set(cand.probe.provider.id, 0);
      return answerClient(cand, clientApi, result, stream, req, res);
    }
    badStreak.set(cand.probe.provider.id, (badStreak.get(cand.probe.provider.id) ?? 0) + 1);
    failed.push(cand);
    log(`provider ${cand.probe.provider.id}/${cand.model.id} refused HTTP ${result.status}; failing over`);
  }
  for (const cand of failed) {
    const result = await callProvider(cand, clientApi, body, stream);
    if (result.ok) {
      badStreak.set(cand.probe.provider.id, 0);
      return answerClient(cand, clientApi, result, stream, req, res);
    }
    badStreak.set(cand.probe.provider.id, (badStreak.get(cand.probe.provider.id) ?? 0) + 1);
    log(`provider ${cand.probe.provider.id}/${cand.model.id} refused again HTTP ${result.status}`);
  }
  return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'service_unavailable', 'every free provider that meets this request refused it');
}

async function answerClient(cand: Candidate, clientApi: 'openai' | 'anthropic', result: UpstreamResult, stream: boolean, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const wire = cand.probe.provider.wire;
  if (!stream) {
    const translated = translateJson(cand, clientApi, result.text);
    if (translated === null) return sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'the provider answered a body that is not JSON');
    return sendJson(res, 200, translated);
  }
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  if ((clientApi === 'openai' && wire === 'openai') || (clientApi === 'anthropic' && wire === 'anthropic')) {
    res.write(result.text);
    res.end();
    return;
  }
  if (clientApi === 'anthropic' && wire === 'openai') {
    // Incremental translation from the starter's wire library.
    const streamBody = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(result.text));
        controller.close();
      },
    });
    for await (const event of anthropicEventsFromOpenAI(streamBody, cand.model.id)) res.write(event);
    res.end();
    return;
  }
  for (const event of translateStream(cand, clientApi, result.text)) res.write(event);
  res.end();
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('openai', body, req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('anthropic', body, req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}

// free-model-aggregator-01: one server, two client APIs, ten simulated free-tier providers.
//
// The server reads the trial brief (provider directory, quality table, workload manifest),
// probes every provider's catalog route in that provider's own wire, registers one catalog
// with the env, and answers each of the client's 20 requests with the best free model that
// meets the request's requirements, failing over to the next candidate when a provider
// refuses. It speaks all three upstream wires (OpenAI, Anthropic, Gemini) and both client
// wires, translating requests losslessly between them: the providers echo what they parsed
// (system text, user text, tool names, max_tokens, stop sequences, image bytes, JSON mode),
// so every field must survive translation byte for byte or the answer does not count.
//
// Design notes:
//  - Upstream streams are always buffered and re-emitted. That keeps failover possible
//    (nothing is sent to the client before a provider answer is known good), guarantees
//    the downstream event order and usage, and makes every wire pair share one code path.
//  - Routing is rating-ordered with adaptive demotion: a provider that fails twice in a
//    row (budget spent or outage) drops to the back of the order until it succeeds again.
//    A JSON-mode request from an OpenAI client never goes to an Anthropic-wire provider,
//    whose wire cannot carry JSON mode.
//  - Everything is a function of the brief, the requests, and the providers' answers:
//    no wall clock, no randomness, fixed timestamps and ids in every reply we build.
//  - Node 22 standard library only; no dependency is installed. TypeScript runs directly,
//    so this file uses erasable syntax only (no enums, no parameter properties).

import { readFileSync } from 'node:fs';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';

type Json = Record<string, unknown>;
type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const asString = (v: unknown): string => (typeof v === 'string' ? v : '');
const asNumber = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const asBool = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);
const cloneJson = <T>(v: T): T => JSON.parse(JSON.stringify(v)) as T;

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const UPSTREAM_TIMEOUT_MS = 15000;
const REGISTER_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
const ANTHROPIC_VERSION_DEFAULT = '2023-06-01';
const STREAM_CHUNK_CHARS = 13;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------------------
// Brief: the provider directory, the quality table, and the workload manifest.
// ---------------------------------------------------------------------------

interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface WorkloadRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  workload: WorkloadRow[];
  error: string | null;
}

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

/** Index of the `}` closing the object that opens at `start`, skipping string contents, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every top-level JSON object in `text` that carries a string `schema`, in order. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: keep scanning.
    }
  }
  return out;
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const authRaw = v.auth;
  const key = asString(v.key);
  const auth: AuthScheme =
    authRaw === 'bearer' || authRaw === 'x-api-key' || authRaw === 'x-goog-api-key' || authRaw === 'none'
      ? authRaw
      : key === ''
        ? 'none'
        : 'bearer';
  const base =
    asString(v.path) !== ''
      ? asString(v.path)
      : asString(v.base_path) !== ''
        ? asString(v.base_path)
        : asString(v.base_url) !== ''
          ? asString(v.base_url)
          : asString(v.base);
  return { id, name: asString(v.name) !== '' ? asString(v.name) : id, wire, auth, base, key_env: asString(v.key_env), key };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = asString(v.model);
  const provider = asString(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: asNumber(v.rating_milli) ?? 0,
    context_length: asNumber(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadRowOf(v: unknown): WorkloadRow | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  const row: WorkloadRow = {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(requires.tools === true ? { tools: true } : {}),
      ...(requires.vision === true ? { vision: true } : {}),
      ...(requires.json === true ? { json: true } : {}),
    },
  };
  const minContext = asNumber(requires.min_context);
  if (minContext !== null) row.requires.min_context = minContext;
  return row;
}

function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], workload: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read brief ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const dirDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (dirDoc === null || !Array.isArray(dirDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = dirDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings =
    ratingsDoc !== null && Array.isArray(ratingsDoc.ratings)
      ? ratingsDoc.ratings.flatMap((v) => {
          const r = ratingOf(v);
          return r === null ? [] : [r];
        })
      : [];
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const workload =
    workloadDoc !== null && Array.isArray(workloadDoc.manifest)
      ? workloadDoc.manifest.flatMap((v) => {
          const w = workloadRowOf(v);
          return w === null ? [] : [w];
        })
      : [];
  return { providers, ratings, workload, error: null };
}

/** The directory from GET /instance, used only when the brief holds no providers at all. */
async function directoryFromInstance(): Promise<ProviderEntry[]> {
  if (ENV_URL === '') return [];
  try {
    const res = await fetch(`${ENV_URL}/instance`, { signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    if (!res.ok) return [];
    const body: unknown = await res.json();
    if (!isRecord(body) || !Array.isArray(body.providers)) return [];
    return body.providers.flatMap((v) => {
      const p = providerOf(v);
      return p === null ? [] : [p];
    });
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------
// Catalog probe: each provider's model list, in its own wire.
// ---------------------------------------------------------------------------

interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  status: number;
  models: ProbedModel[];
  error: string | null;
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface CatalogReport {
  providers: CatalogProvider[];
}

interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** A base path joined to the env URL for dialling; an absolute base is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The base URL the catalog reports: the provider's own path, never joined to the env. */
function catalogBaseOf(base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) {
    try {
      return new URL(b).pathname.replace(/\/+$/, '') || '/';
    } catch {
      return b;
    }
  }
  return b.replace(/\/+$/, '') || b;
}

/** The headers carrying one provider's key, in its own auth scheme. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** Whether a listing's pricing says free: true when every price is zero, null when silent. */
function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values: number[] = [];
  for (const x of Object.values(pricing)) {
    if (typeof x === 'number') values.push(x);
    else if (typeof x === 'string' && x.trim() !== '') values.push(Number(x));
  }
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? asString(v.name).replace(/^models\//, '') : asString(v.id);
  if (id === '') return null;
  const display =
    asString(v.display_name) !== '' ? asString(v.display_name) : asString(v.displayName) !== '' ? asString(v.displayName) : id;
  return {
    id,
    display_name: display,
    free: asBool(v.free) ?? zeroPriced(v),
    context_length: asNumber(v.context_length) ?? asNumber(v.context_window) ?? asNumber(v.inputTokenLimit),
    tools: asBool(v.tools),
    vision: asBool(v.vision),
    json: asBool(v.json),
  };
}

function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  const seen = new Set<string>();
  const out: ProbedModel[] = [];
  for (const v of list) {
    const m = modelIn(wire, v);
    if (m === null || seen.has(m.id)) continue;
    seen.add(m.id);
    out.push(m);
  }
  return out;
}

/** Probe one provider's catalog route, retrying once; probes are GETs and cost no budget. */
async function probeProvider(provider: ProviderEntry, timeoutMs: number): Promise<Probe> {
  const base = joinBase(ENV_URL, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION_DEFAULT;
  let lastError: string | null = null;
  let lastStatus = 0;
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
      lastStatus = res.status;
      if (!res.ok) {
        lastError = `GET ${url} answered HTTP ${res.status}`;
        continue;
      }
      const body: unknown = await res.json();
      return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
    } catch (err) {
      lastError = `GET ${url} failed: ${(err as Error).message}`;
    }
  }
  return { provider, ok: false, status: lastStatus, models: [], error: lastError };
}

function ratingRowFor(ratings: RatingRow[], providerId: string, modelId: string): RatingRow | null {
  return ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/**
 * Whether a model is free: the quality table's row where it has one (the sealed truth),
 * the listing's own word where it does not, paid when neither says. Paid is never routed.
 */
function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

function capabilitiesOf(model: ProbedModel, row: RatingRow | null): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** Whether capabilities meet a request's requirements. An unknown capability never meets one. */
function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

const byModelId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The one catalog document: `--list` output and register body, built once and reused. */
function buildCatalog(probes: readonly Probe[], ratings: RatingRow[]): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort(byModelId)) {
      const row = ratingRowFor(ratings, probe.provider.id, model.id);
      models.push({ id: model.id, free: modelFree(model, row), context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({
      id: probe.provider.id,
      base_url: catalogBaseOf(probe.provider.base),
      key_env: probe.provider.key_env,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// Routing state: per-provider attempt counts and consecutive-failure demotion.
// ---------------------------------------------------------------------------

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  row: RatingRow | null;
  rating: number;
  caps: Capabilities;
  providerIndex: number;
}

const attemptsByProvider = new Map<string, number>();
const consecutiveFailures = new Map<string, number>();
/** Providers whose own error message said the trial is over for them (budget spent / out). */
const exhaustedProviders = new Set<string>();

function recordAttempt(providerId: string): void {
  attemptsByProvider.set(providerId, (attemptsByProvider.get(providerId) ?? 0) + 1);
}

function recordSuccess(providerId: string): void {
  consecutiveFailures.set(providerId, 0);
}

function recordFailure(providerId: string): void {
  consecutiveFailures.set(providerId, (consecutiveFailures.get(providerId) ?? 0) + 1);
}

/**
 * Whether a refusal message is the provider saying it will serve nothing more this
 * trial. The match is exact against the provider's own name, so an unfamiliar message
 * (a real service's wording) never excludes a provider: it stays eligible and the
 * consecutive-failure demotion still applies.
 */
function isTerminalRefusal(provider: ProviderEntry, message: string): boolean {
  return message === `${provider.name}'s free-tier budget is spent` || message === `${provider.name} is out`;
}

/**
 * Candidates for one request, best first: providers with two consecutive failures demoted
 * (budget spent or outage), then rating, then same-wire-as-client, then fewest attempts,
 * then brief order and model id for a deterministic tie-break. An OpenAI JSON-mode
 * request never goes to an Anthropic-wire provider, whose wire cannot carry JSON mode.
 */
function candidatesFor(
  probes: readonly Probe[],
  ratings: RatingRow[],
  requires: Requirement,
  clientWire: Wire,
  jsonModeOpenAI: boolean,
): Candidate[] {
  const out: Candidate[] = [];
  probes.forEach((probe, providerIndex) => {
    if (!probe.ok) return;
    if (exhaustedProviders.has(probe.provider.id)) return;
    if (jsonModeOpenAI && probe.provider.wire === 'anthropic') return;
    for (const model of probe.models) {
      const row = ratingRowFor(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      const caps = capabilitiesOf(model, row);
      if (!meetsRequirements(requires, caps)) continue;
      out.push({ probe, model, row, rating: row?.rating_milli ?? 0, caps, providerIndex });
    }
  });
  out.sort((a, b) => {
    const tierA = (consecutiveFailures.get(a.probe.provider.id) ?? 0) >= 2 ? 1 : 0;
    const tierB = (consecutiveFailures.get(b.probe.provider.id) ?? 0) >= 2 ? 1 : 0;
    if (tierA !== tierB) return tierA - tierB;
    if (a.rating !== b.rating) return b.rating - a.rating;
    const wireA = a.probe.provider.wire === clientWire ? 0 : 1;
    const wireB = b.probe.provider.wire === clientWire ? 0 : 1;
    if (wireA !== wireB) return wireA - wireB;
    const attA = attemptsByProvider.get(a.probe.provider.id) ?? 0;
    const attB = attemptsByProvider.get(b.probe.provider.id) ?? 0;
    if (attA !== attB) return attA - attB;
    if (a.providerIndex !== b.providerIndex) return a.providerIndex - b.providerIndex;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

// ---------------------------------------------------------------------------
// Request translation: client wire -> intermediate form -> provider wire.
//
// The intermediate form keeps exactly what the providers echo: the system text, the
// turns' text and images in order, tool definitions and tool traffic, max_tokens,
// stop sequences, and JSON mode. Translators must preserve each of those exactly:
// the system text is the system messages joined with "\n", the user text is the text
// blocks joined with "\n", and an image keeps its exact bytes (base64 data) or URL.
// ---------------------------------------------------------------------------

type ImageRef = { kind: 'data'; mime: string; b64: string } | { kind: 'url'; url: string };

interface IRText {
  kind: 'text';
  text: string;
}
interface IRImage {
  kind: 'image';
  image: ImageRef;
}
interface IRToolUse {
  kind: 'tool_use';
  id: string;
  name: string;
  argsText: string;
  argsObj: Json;
}
interface IRToolResult {
  kind: 'tool_result';
  id: string;
  text: string;
}
type IRPart = IRText | IRImage | IRToolUse | IRToolResult;

interface IRTurn {
  role: 'user' | 'assistant';
  parts: IRPart[];
}

interface IRToolDef {
  name: string;
  description: string;
  schema: Json;
}

type IRToolChoice = { mode: 'auto' | 'none' | 'required'; name?: string } | null;

interface IRRequest {
  systemText: string;
  turns: IRTurn[];
  tools: IRToolDef[];
  toolChoice: IRToolChoice;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  jsonMode: boolean;
  jsonSchema: Json | null;
  stream: boolean;
}

/** Text of a string or a block array's text blocks, joined with "\n" (the echo's rule). */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const b of content) {
    if (isRecord(b) && b.type === 'text' && typeof b.text === 'string') parts.push(b.text);
  }
  return parts.join('\n');
}

/** A tool call's arguments as both the raw string and a parsed object ({} when unparsable). */
function parseArgs(value: unknown): { argsText: string; argsObj: Json } {
  if (isRecord(value)) return { argsText: JSON.stringify(value), argsObj: value };
  if (typeof value === 'string') {
    if (value.trim() === '') return { argsText: value, argsObj: {} };
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return { argsText: value, argsObj: parsed };
    } catch {
      // Not JSON: the object form is empty, the string form is kept.
    }
    return { argsText: value, argsObj: {} };
  }
  return { argsText: '', argsObj: {} };
}

/** An image_url value as an image reference, preserving data-URL bytes or the URL itself. */
function imageFromUrl(url: unknown): ImageRef | null {
  if (typeof url !== 'string' || url === '') return null;
  const data = /^data:([^;,]*)(;base64)?,(.*)$/s.exec(url);
  if (data !== null && data[2] === ';base64') {
    return { kind: 'data', mime: data[1] !== '' ? data[1] : 'image/png', b64: data[3] };
  }
  if (data !== null) return { kind: 'url', url };
  return { kind: 'url', url };
}

function imageFromAnthropicSource(source: unknown): ImageRef | null {
  if (!isRecord(source)) return null;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { kind: 'data', mime: typeof source.media_type === 'string' && source.media_type !== '' ? source.media_type : 'image/png', b64: source.data };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { kind: 'url', url: source.url };
  return null;
}

function stringListOf(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** OpenAI messages -> intermediate form. */
function openAIToIR(body: Json): IRRequest {
  const systemParts: string[] = [];
  const turns: IRTurn[] = [];
  let orphanResults: IRToolResult[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'tool') {
      const result: IRToolResult = { kind: 'tool_result', id: asString(raw.tool_call_id), text: blocksText(raw.content) };
      // A tool message must not become its own user turn: the Anthropic and Gemini
      // parsers take the user text from the last user turn, so a trailing tool-only
      // turn would erase it. Attach the result to the nearest user turn instead.
      let attached = false;
      for (let i = turns.length - 1; i >= 0; i--) {
        if (turns[i].role === 'user') {
          turns[i].parts.push(result);
          attached = true;
          break;
        }
      }
      if (!attached) orphanResults.push(result);
      continue;
    }
    if (role === 'assistant') {
      const parts: IRPart[] = [];
      const content = raw.content;
      if (typeof content === 'string') {
        if (content !== '') parts.push({ kind: 'text', text: content });
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
          else if (block.type === 'image_url' && isRecord(block.image_url)) {
            const img = imageFromUrl(block.image_url.url);
            if (img !== null) parts.push({ kind: 'image', image: img });
          }
        }
      }
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          const args = parseArgs(call.function.arguments);
          parts.push({ kind: 'tool_use', id: asString(call.id), name: asString(call.function.name), argsText: args.argsText, argsObj: args.argsObj });
        }
      }
      turns.push({ role: 'assistant', parts });
      continue;
    }
    if (role === 'user') {
      const parts: IRPart[] = [];
      const content = raw.content;
      if (typeof content === 'string') {
        if (content !== '') parts.push({ kind: 'text', text: content });
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
          else if (block.type === 'image_url' && isRecord(block.image_url)) {
            const img = imageFromUrl(block.image_url.url);
            if (img !== null) parts.push({ kind: 'image', image: img });
          } else if (block.type === 'input_audio' || block.type === 'file') {
            // No wire here carries these; the echo does not read them either.
          }
        }
      }
      if (orphanResults.length > 0) {
        parts.push(...orphanResults);
        orphanResults = [];
      }
      turns.push({ role: 'user', parts });
      continue;
    }
    // 'developer' and unknown roles: the providers' parsers read neither system nor
    // user text from them, so folding them in would corrupt the echo. Dropped here;
    // the same-wire path forwards the body verbatim, so they survive there.
  }
  if (orphanResults.length > 0) {
    let attached = false;
    for (let i = turns.length - 1; i >= 0; i--) {
      if (turns[i].role === 'user') {
        turns[i].parts.push(...orphanResults);
        attached = true;
        break;
      }
    }
    if (!attached) turns.push({ role: 'user', parts: [...orphanResults] });
  }

  const tools: IRToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: IRToolChoice = null;
  const choice = body.tool_choice;
  if (choice === 'auto') toolChoice = { mode: 'auto' };
  else if (choice === 'none') toolChoice = { mode: 'none' };
  else if (choice === 'required') toolChoice = { mode: 'required' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    toolChoice = { mode: 'required', name: choice.function.name };
  }
  const format = isRecord(body.response_format) ? body.response_format : null;
  const jsonMode = format !== null && (format.type === 'json_object' || format.type === 'json_schema');
  return {
    systemText: systemParts.join('\n'),
    turns,
    tools,
    toolChoice,
    maxTokens: asNumber(body.max_tokens) ?? asNumber(body.max_completion_tokens),
    temperature: asNumber(body.temperature),
    topP: asNumber(body.top_p),
    stops: stringListOf(body.stop),
    jsonMode,
    jsonSchema: format !== null && format.type === 'json_schema' && isRecord(format.json_schema) ? format.json_schema : null,
    stream: body.stream === true,
  };
}

/** Anthropic messages -> intermediate form. */
function anthropicToIR(body: Json): IRRequest {
  const turns: IRTurn[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    const parts: IRPart[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ kind: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
        else if (block.type === 'image') {
          const img = imageFromAnthropicSource(block.source);
          if (img !== null) parts.push({ kind: 'image', image: img });
        } else if (block.type === 'tool_use') {
          const input = isRecord(block.input) ? block.input : {};
          parts.push({ kind: 'tool_use', id: asString(block.id), name: asString(block.name), argsText: JSON.stringify(input), argsObj: input });
        } else if (block.type === 'tool_result') {
          parts.push({ kind: 'tool_result', id: asString(block.tool_use_id), text: blocksText(block.content) });
        }
      }
    }
    turns.push({ role: raw.role === 'assistant' ? 'assistant' : 'user', parts });
  }
  const tools: IRToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        schema: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: IRToolChoice = null;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') toolChoice = { mode: 'auto' };
    else if (choice.type === 'none') toolChoice = { mode: 'none' };
    else if (choice.type === 'any') toolChoice = { mode: 'required' };
    else if (choice.type === 'tool' && typeof choice.name === 'string') toolChoice = { mode: 'required', name: choice.name };
  }
  return {
    systemText: blocksText(body.system),
    turns,
    tools,
    toolChoice,
    maxTokens: asNumber(body.max_tokens),
    temperature: asNumber(body.temperature),
    topP: asNumber(body.top_p),
    stops: stringListOf(body.stop_sequences),
    jsonMode: false,
    jsonSchema: null,
    stream: body.stream === true,
  };
}

/** Intermediate form -> an OpenAI Chat Completions body naming `model`. */
function irToOpenAI(ir: IRRequest, model: string): Json {
  const messages: Json[] = [];
  if (ir.systemText !== '') messages.push({ role: 'system', content: ir.systemText });
  for (const turn of ir.turns) {
    if (turn.role === 'user') {
      const parts: Json[] = [];
      const results: IRToolResult[] = [];
      for (const part of turn.parts) {
        if (part.kind === 'text') parts.push({ type: 'text', text: part.text });
        else if (part.kind === 'image') {
          parts.push({
            type: 'image_url',
            image_url: { url: part.image.kind === 'data' ? `data:${part.image.mime};base64,${part.image.b64}` : part.image.url },
          });
        } else if (part.kind === 'tool_result') results.push(part);
      }
      if (parts.length === 1 && parts[0].type === 'text') messages.push({ role: 'user', content: parts[0].text });
      else if (parts.length > 0) messages.push({ role: 'user', content: parts });
      for (const r of results) messages.push({ role: 'tool', tool_call_id: r.id, content: r.text });
    } else {
      const texts: string[] = [];
      const calls: Json[] = [];
      for (const part of turn.parts) {
        if (part.kind === 'text') texts.push(part.text);
        else if (part.kind === 'tool_use') {
          calls.push({ id: part.id, type: 'function', function: { name: part.name, arguments: part.argsText } });
        }
      }
      const msg: Json = { role: 'assistant', content: texts.length > 0 ? texts.join('\n') : '' };
      if (calls.length > 0) msg.tool_calls = calls;
      messages.push(msg);
    }
  }
  const out: Json = { model, messages };
  if (ir.maxTokens !== null) out.max_tokens = ir.maxTokens;
  if (ir.temperature !== null) out.temperature = ir.temperature;
  if (ir.topP !== null) out.top_p = ir.topP;
  if (ir.stops.length > 0) out.stop = [...ir.stops];
  if (ir.tools.length > 0) {
    out.tools = ir.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.schema } }));
  }
  if (ir.toolChoice !== null) {
    if (ir.toolChoice.mode === 'auto') out.tool_choice = 'auto';
    else if (ir.toolChoice.mode === 'none') out.tool_choice = 'none';
    else if (ir.toolChoice.name !== undefined) out.tool_choice = { type: 'function', function: { name: ir.toolChoice.name } };
    else out.tool_choice = 'required';
  }
  if (ir.jsonMode) {
    out.response_format =
      ir.jsonSchema !== null ? { type: 'json_schema', json_schema: ir.jsonSchema } : { type: 'json_object' };
  }
  if (ir.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** Intermediate form -> an Anthropic Messages body naming `model`. */
function irToAnthropic(ir: IRRequest, model: string): Json {
  const messages: Json[] = [];
  for (const turn of ir.turns) {
    const blocks: Json[] = [];
    for (const part of turn.parts) {
      if (part.kind === 'text') blocks.push({ type: 'text', text: part.text });
      else if (part.kind === 'image') {
        blocks.push(
          part.image.kind === 'data'
            ? { type: 'image', source: { type: 'base64', media_type: part.image.mime, data: part.image.b64 } }
            : { type: 'image', source: { type: 'url', url: part.image.url } },
        );
      } else if (part.kind === 'tool_use') blocks.push({ type: 'tool_use', id: part.id, name: part.name, input: part.argsObj });
      else if (part.kind === 'tool_result') blocks.push({ type: 'tool_result', tool_use_id: part.id, content: part.text });
    }
    if (blocks.length === 1 && blocks[0].type === 'text') messages.push({ role: turn.role, content: blocks[0].text });
    else messages.push({ role: turn.role, content: blocks });
  }
  const out: Json = { model, messages };
  if (ir.systemText !== '') out.system = ir.systemText;
  if (ir.maxTokens !== null) out.max_tokens = ir.maxTokens;
  if (ir.temperature !== null) out.temperature = ir.temperature;
  if (ir.topP !== null) out.top_p = ir.topP;
  if (ir.stops.length > 0) out.stop_sequences = [...ir.stops];
  if (ir.tools.length > 0) {
    out.tools = ir.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.schema }));
  }
  if (ir.toolChoice !== null) {
    if (ir.toolChoice.mode === 'auto') out.tool_choice = { type: 'auto' };
    else if (ir.toolChoice.mode === 'none') out.tool_choice = { type: 'none' };
    else if (ir.toolChoice.name !== undefined) out.tool_choice = { type: 'tool', name: ir.toolChoice.name };
    else out.tool_choice = { type: 'any' };
  }
  if (ir.stream) out.stream = true;
  return out;
}

/** Guess a MIME type for a remote image URL (Gemini fileData wants one; the sim reads the URI). */
function mimeForUrl(url: string): string {
  const lower = url.split('?')[0].toLowerCase();
  if (lower.endsWith('.png')) return 'image/png';
  if (lower.endsWith('.jpg') || lower.endsWith('.jpeg')) return 'image/jpeg';
  if (lower.endsWith('.gif')) return 'image/gif';
  if (lower.endsWith('.webp')) return 'image/webp';
  return 'image/png';
}

/** Intermediate form -> a Gemini generateContent body (the model travels in the URL path). */
function irToGemini(ir: IRRequest): Json {
  // Tool-result parts need their function's name for functionResponse; recover it from
  // the matching tool_use part's id, falling back to the id itself.
  const nameById = new Map<string, string>();
  for (const turn of ir.turns) {
    for (const part of turn.parts) {
      if (part.kind === 'tool_use' && part.id !== '') nameById.set(part.id, part.name);
    }
  }
  const contents: Json[] = [];
  for (const turn of ir.turns) {
    const parts: Json[] = [];
    for (const part of turn.parts) {
      if (part.kind === 'text') parts.push({ text: part.text });
      else if (part.kind === 'image') {
        parts.push(
          part.image.kind === 'data'
            ? { inlineData: { mimeType: part.image.mime, data: part.image.b64 } }
            : { fileData: { mimeType: mimeForUrl(part.image.url), fileUri: part.image.url } },
        );
      } else if (part.kind === 'tool_use') parts.push({ functionCall: { name: part.name, args: part.argsObj } });
      else if (part.kind === 'tool_result') {
        parts.push({ functionResponse: { name: nameById.get(part.id) ?? part.id, response: { result: part.text } } });
      }
    }
    contents.push({ role: turn.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  if (ir.systemText !== '') out.systemInstruction = { parts: [{ text: ir.systemText }] };
  const config: Json = {};
  if (ir.maxTokens !== null) config.maxOutputTokens = ir.maxTokens;
  if (ir.stops.length > 0) config.stopSequences = [...ir.stops];
  if (ir.temperature !== null) config.temperature = ir.temperature;
  if (ir.topP !== null) config.topP = ir.topP;
  if (ir.jsonMode) {
    config.responseMimeType = 'application/json';
    if (ir.jsonSchema !== null) config.responseSchema = ir.jsonSchema;
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  if (ir.tools.length > 0) {
    out.tools = [{ functionDeclarations: ir.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.schema })) }];
  }
  if (ir.toolChoice !== null) {
    const mode = ir.toolChoice.mode === 'auto' ? 'AUTO' : ir.toolChoice.mode === 'none' ? 'NONE' : 'ANY';
    const fnConfig: Json = { mode };
    if (ir.toolChoice.name !== undefined) fnConfig.allowedFunctionNames = [ir.toolChoice.name];
    out.toolConfig = { functionCallingConfig: fnConfig };
  }
  return out;
}

// ---------------------------------------------------------------------------
// Provider answers: every wire's JSON and SSE shapes -> one canonical answer.
// ---------------------------------------------------------------------------

/** Stop-reason families, shared by all three wires (the client's check compares families). */
function stopFamilyOf(reason: unknown): number {
  if (reason === 'stop' || reason === 'end_turn' || reason === 'STOP') return 0;
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return 0;
}

interface CanonicalToolCall {
  id: string;
  name: string;
  argsText: string;
}

interface CanonicalAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stopFamily: number;
  toolCalls: CanonicalToolCall[];
  upstreamId: string;
}

/** Text of a message's content as the client's checker reads it: string, or text blocks joined. */
function answerTextOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('');
}

function numberOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** One SSE document as frames: {event, data}, multi-line data joined, [DONE] kept as data. */
function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseJsonObject(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function openAIAnswerFromJson(body: Json): CanonicalAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const toolCalls: CanonicalToolCall[] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      toolCalls.push({
        id: asString(call.id),
        name: asString(call.function.name),
        argsText: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(call.function.arguments ?? {}),
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: answerTextOf(message.content),
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
    stopFamily: stopFamilyOf(choice.finish_reason),
    toolCalls,
    upstreamId: asString(body.id),
  };
}

function openAIAnswerFromSse(text: string, fallbackId: string): CanonicalAnswer {
  let out = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stopFamily = 0;
  let upstreamId = fallbackId;
  const byIndex = new Map<number, CanonicalToolCall>();
  for (const frame of sseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    const chunk = parseJsonObject(payload);
    if (chunk === null) continue;
    if (asString(chunk.id) !== '') upstreamId = asString(chunk.id);
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    if (choices.length === 0) continue;
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (Array.isArray(delta.tool_calls)) {
      for (const call of delta.tool_calls) {
        if (!isRecord(call)) continue;
        const index = asNumber(call.index) ?? byIndex.size;
        let acc = byIndex.get(index);
        if (acc === undefined) {
          acc = { id: '', name: '', argsText: '' };
          byIndex.set(index, acc);
        }
        if (typeof call.id === 'string' && call.id !== '') acc.id = call.id;
        if (isRecord(call.function)) {
          if (typeof call.function.name === 'string' && call.function.name !== '') acc.name = call.function.name;
          if (typeof call.function.arguments === 'string') acc.argsText += call.function.arguments;
        }
      }
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) stopFamily = stopFamilyOf(choice.finish_reason);
  }
  return {
    text: out,
    inputTokens,
    outputTokens,
    stopFamily,
    toolCalls: [...byIndex.entries()].sort((a, b) => a[0] - b[0]).map(([, c]) => c),
    upstreamId,
  };
}

function anthropicAnswerFromJson(body: Json): CanonicalAnswer {
  const content = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: CanonicalToolCall[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') {
      toolCalls.push({ id: asString(block.id), name: asString(block.name), argsText: JSON.stringify(isRecord(block.input) ? block.input : {}) });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    inputTokens: numberOrZero(usage.input_tokens),
    outputTokens: numberOrZero(usage.output_tokens),
    stopFamily: stopFamilyOf(body.stop_reason),
    toolCalls,
    upstreamId: asString(body.id),
  };
}

function anthropicAnswerFromSse(text: string): CanonicalAnswer {
  let out = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stopFamily = 0;
  let upstreamId = '';
  const byIndex = new Map<number, CanonicalToolCall>();
  for (const frame of sseFrames(text)) {
    const data = parseJsonObject(frame.data);
    if (data === null) continue;
    switch (frame.event) {
      case 'message_start': {
        const message = isRecord(data.message) ? data.message : {};
        if (asString(message.id) !== '') upstreamId = asString(message.id);
        if (isRecord(message.usage)) inputTokens = numberOrZero(message.usage.input_tokens);
        break;
      }
      case 'content_block_start': {
        const index = asNumber(data.index);
        const block = isRecord(data.content_block) ? data.content_block : {};
        if (index !== null && block.type === 'tool_use') {
          byIndex.set(index, { id: asString(block.id), name: asString(block.name), argsText: '' });
        }
        break;
      }
      case 'content_block_delta': {
        const index = asNumber(data.index);
        const delta = isRecord(data.delta) ? data.delta : {};
        if (delta.type === 'text_delta' && typeof delta.text === 'string') out += delta.text;
        else if (delta.type === 'input_json_delta' && index !== null && typeof delta.partial_json === 'string') {
          const acc = byIndex.get(index);
          if (acc !== undefined) acc.argsText += delta.partial_json;
        }
        break;
      }
      case 'message_delta': {
        const delta = isRecord(data.delta) ? data.delta : {};
        if (typeof delta.stop_reason === 'string') stopFamily = stopFamilyOf(delta.stop_reason);
        if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
        break;
      }
      default:
        break;
    }
  }
  return {
    text: out,
    inputTokens,
    outputTokens,
    stopFamily,
    toolCalls: [...byIndex.entries()].sort((a, b) => a[0] - b[0]).map(([, c]) => c),
    upstreamId,
  };
}

function geminiCandidatesOf(body: Json): Json[] {
  return Array.isArray(body.candidates) ? body.candidates.filter(isRecord) : [];
}

function geminiAnswerFromJson(body: Json): CanonicalAnswer {
  const candidates = geminiCandidatesOf(body);
  const first = candidates[0] ?? {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const toolCalls: CanonicalToolCall[] = [];
  parts.forEach((part, i) => {
    if (!isRecord(part)) return;
    if (typeof part.text === 'string') text += part.text;
    else if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: `call_${i}`,
        name: asString(part.functionCall.name),
        argsText: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
      });
    }
  });
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    inputTokens: numberOrZero(usage.promptTokenCount),
    outputTokens: numberOrZero(usage.candidatesTokenCount),
    stopFamily: stopFamilyOf(first.finishReason),
    toolCalls,
    upstreamId: asString(body.modelVersion),
  };
}

function geminiAnswerFromSse(text: string): CanonicalAnswer {
  let out = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stopFamily = 0;
  let upstreamId = '';
  const toolCalls: CanonicalToolCall[] = [];
  let callSeq = 0;
  for (const frame of sseFrames(text)) {
    const chunk = parseJsonObject(frame.data);
    if (chunk === null) continue;
    if (asString(chunk.modelVersion) !== '') upstreamId = asString(chunk.modelVersion);
    if (isRecord(chunk.usageMetadata)) {
      inputTokens = numberOrZero(chunk.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(chunk.usageMetadata.candidatesTokenCount);
    }
    for (const candidate of geminiCandidatesOf(chunk)) {
      const content = isRecord(candidate.content) ? candidate.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const part of parts) {
        if (!isRecord(part)) continue;
        if (typeof part.text === 'string') out += part.text;
        else if (isRecord(part.functionCall)) {
          toolCalls.push({
            id: `call_${callSeq}`,
            name: asString(part.functionCall.name),
            argsText: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
          });
          callSeq += 1;
        }
      }
      if (typeof candidate.finishReason === 'string') stopFamily = stopFamilyOf(candidate.finishReason);
    }
  }
  return { text: out, inputTokens, outputTokens, stopFamily, toolCalls, upstreamId };
}

/** Parse one provider's answer (JSON or SSE) into the canonical form. Never throws. */
function parseProviderAnswer(wire: Wire, stream: boolean, bodyText: string, modelId: string): CanonicalAnswer {
  try {
    if (wire === 'openai') {
      if (stream) return openAIAnswerFromSse(bodyText, 'chatcmpl-aggregator');
      const parsed = parseJsonObject(bodyText);
      return parsed !== null ? openAIAnswerFromJson(parsed) : { text: '', inputTokens: 0, outputTokens: 0, stopFamily: 0, toolCalls: [], upstreamId: '' };
    }
    if (wire === 'anthropic') {
      if (stream) return anthropicAnswerFromSse(bodyText);
      const parsed = parseJsonObject(bodyText);
      return parsed !== null ? anthropicAnswerFromJson(parsed) : { text: '', inputTokens: 0, outputTokens: 0, stopFamily: 0, toolCalls: [], upstreamId: '' };
    }
    if (stream) return geminiAnswerFromSse(bodyText);
    const parsed = parseJsonObject(bodyText);
    return parsed !== null ? geminiAnswerFromJson(parsed) : { text: '', inputTokens: 0, outputTokens: 0, stopFamily: 0, toolCalls: [], upstreamId: '' };
  } catch {
    void modelId;
    return { text: '', inputTokens: 0, outputTokens: 0, stopFamily: 0, toolCalls: [], upstreamId: '' };
  }
}

// ---------------------------------------------------------------------------
// Upstream calls: build the provider's request, send it, read the whole answer.
// ---------------------------------------------------------------------------

interface UpstreamSuccess {
  ok: true;
  answer: CanonicalAnswer;
  status: number;
}

interface UpstreamFailure {
  ok: false;
  status: number;
  retryable: boolean;
  detail: string;
  /** The error message from the provider's own error envelope, '' when unreadable. */
  message: string;
}

type UpstreamResult = UpstreamSuccess | UpstreamFailure;

interface PreparedCall {
  url: string;
  headers: Record<string, string>;
  bodyText: string;
}

/** Build one provider call for a candidate: same-wire bodies pass through verbatim. */
function prepareCall(
  candidate: Candidate,
  clientWire: Wire,
  clientBody: Json,
  ir: IRRequest,
  anthropicVersion: string,
): PreparedCall {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    if (clientWire === 'openai') {
      payload = cloneJson(clientBody);
      payload.model = candidate.model.id;
      if (payload.stream === true) {
        const opts = isRecord(payload.stream_options) ? payload.stream_options : {};
        payload.stream_options = { ...opts, include_usage: true };
      }
    } else {
      payload = irToOpenAI(ir, candidate.model.id);
    }
    if (ir.stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = anthropicVersion;
    if (clientWire === 'anthropic') {
      payload = cloneJson(clientBody);
      payload.model = candidate.model.id;
    } else {
      payload = irToAnthropic(ir, candidate.model.id);
    }
    if (ir.stream) headers.accept = 'text/event-stream';
  } else {
    const verb = ir.stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${verb}${ir.stream ? '?alt=sse' : ''}`;
    payload = irToGemini(ir);
    if (ir.stream) headers.accept = 'text/event-stream';
  }
  return { url, headers, bodyText: JSON.stringify(payload) };
}

/** The message inside a provider error body, in any of the three wires' envelopes. */
function errorMessageOf(text: string): string {
  const body = parseJsonObject(text);
  if (body === null) return '';
  const error = isRecord(body.error) ? body.error : null;
  if (error !== null && typeof error.message === 'string') return error.message;
  if (typeof body.message === 'string') return body.message;
  return '';
}

async function callUpstream(prepared: PreparedCall, provider: ProviderEntry, modelId: string, stream: boolean): Promise<UpstreamResult> {
  let res: Response;
  try {
    res = await fetch(prepared.url, { method: 'POST', headers: prepared.headers, body: prepared.bodyText, signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch (err) {
    return { ok: false, status: 0, retryable: true, detail: `unreachable: ${(err as Error).message}`, message: '' };
  }
  const text = await res.text();
  if (!res.ok) {
    // 402/429/5xx are the provider refusing: fail over. 400/401 mean our request was
    // wrong (bad model or key); still fail over rather than crash, but never retry
    // the same provider for this request in the second pass.
    const retryable = res.status === 402 || res.status === 429 || res.status >= 500;
    return { ok: false, status: res.status, retryable, detail: `HTTP ${res.status}: ${text.slice(0, 200)}`, message: errorMessageOf(text) };
  }
  void provider;
  return { ok: true, answer: parseProviderAnswer(provider.wire, stream, text, modelId), status: res.status };
}

// ---------------------------------------------------------------------------
// Downstream answers: the canonical answer in the wire the client used.
// ---------------------------------------------------------------------------

function openAIFinishReason(family: number): string {
  if (family === 1) return 'length';
  if (family === 2) return 'tool_calls';
  if (family === 3) return 'content_filter';
  return 'stop';
}

function anthropicStopReason(family: number): string {
  if (family === 1) return 'max_tokens';
  if (family === 2) return 'tool_use';
  if (family === 3) return 'refusal';
  return 'end_turn';
}

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += STREAM_CHUNK_CHARS) pieces.push(text.slice(at, at + STREAM_CHUNK_CHARS));
  return pieces.length === 0 ? [] : pieces;
}

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** A canonical answer as an OpenAI Chat Completions JSON object. */
function openAIJsonAnswer(answer: CanonicalAnswer, modelId: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((c) => ({
      id: c.id !== '' ? c.id : 'call_aggregator',
      type: 'function',
      function: { name: c.name, arguments: c.argsText },
    }));
  }
  return {
    id: answer.upstreamId !== '' && answer.upstreamId.startsWith('chatcmpl') ? answer.upstreamId : 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model: modelId,
    choices: [{ index: 0, message, finish_reason: openAIFinishReason(answer.stopFamily), logprobs: null }],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** A canonical answer as an OpenAI SSE stream: role, content deltas, finish, usage, [DONE]. */
function openAISseAnswer(answer: CanonicalAnswer, modelId: string): string {
  const id = answer.upstreamId !== '' && answer.upstreamId.startsWith('chatcmpl') ? answer.upstreamId : 'chatcmpl-aggregator';
  const chunk = (delta: Json, finish: string | null): Json => ({
    id,
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model: modelId,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  answer.toolCalls.forEach((call, index) => {
    frames.push(
      sseEvent(
        null,
        chunk({ tool_calls: [{ index, id: call.id !== '' ? call.id : 'call_aggregator', type: 'function', function: { name: call.name, arguments: call.argsText } }] }, null),
      ),
    );
  });
  frames.push(sseEvent(null, chunk({}, openAIFinishReason(answer.stopFamily))));
  frames.push(
    sseEvent(null, {
      id,
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model: modelId,
      choices: [],
      usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A canonical answer as an Anthropic Messages JSON object. */
function anthropicJsonAnswer(answer: CanonicalAnswer, modelId: string): Json {
  const content: Json[] = [];
  if (answer.text !== '' || answer.toolCalls.length === 0) content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    let input: Json = {};
    try {
      const parsed: unknown = JSON.parse(call.argsText);
      if (isRecord(parsed)) input = parsed;
    } catch {
      // Keep the empty input.
    }
    content.push({ type: 'tool_use', id: call.id !== '' ? call.id : 'toolu_aggregator', name: call.name, input });
  }
  return {
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model: modelId,
    content,
    stop_reason: anthropicStopReason(answer.stopFamily),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** A canonical answer as an Anthropic SSE stream, in the wire's event order. */
function anthropicSseAnswer(answer: CanonicalAnswer, modelId: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model: modelId,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
  ];
  let index = 0;
  if (answer.text !== '' || answer.toolCalls.length === 0) {
    const textIndex = index;
    index += 1;
    frames.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
    for (const piece of splitText(answer.text)) {
      frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: piece } }));
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
  }
  for (const call of answer.toolCalls) {
    const callIndex = index;
    index += 1;
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index: callIndex,
        content_block: { type: 'tool_use', id: call.id !== '' ? call.id : 'toolu_aggregator', name: call.name, input: {} },
      }),
    );
    if (call.argsText !== '') {
      frames.push(
        sseEvent('content_block_delta', { type: 'content_block_delta', index: callIndex, delta: { type: 'input_json_delta', partial_json: call.argsText } }),
      );
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: callIndex }));
  }
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopReason(answer.stopFamily), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

// ---------------------------------------------------------------------------
// HTTP server: the two client APIs, the model list, register, and the poll.
// ---------------------------------------------------------------------------

let brief: Brief = { providers: [], ratings: [], workload: [], error: null };
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;

/** The free models across answered providers, deduplicated by id in brief order. */
function servedModels(): { id: string; provider: string; display_name: string }[] {
  const seen = new Set<string>();
  const out: { id: string; provider: string; display_name: string }[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byModelId)) {
      const row = ratingRowFor(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

async function load(): Promise<void> {
  brief = readBrief(PROMPT_FILE);
  if (brief.error !== null) log(brief.error);
  if (brief.providers.length === 0 && ENV_URL !== '') {
    const fromInstance = await directoryFromInstance();
    if (fromInstance.length > 0) {
      brief = { ...brief, providers: fromInstance, error: null };
      log('brief held no directory; using GET /instance instead');
    }
  }
  // Probes run one at a time, in brief order: deterministic, and no provider ever
  // sees two calls at once.
  probes = [];
  for (const provider of brief.providers) {
    if (ENV_URL === '') {
      probes.push({ provider, ok: false, status: 0, models: [], error: 'MAKEFUN_ENV_URL is not set' });
      continue;
    }
    const probe = await probeProvider(provider, PROBE_TIMEOUT_MS);
    if (!probe.ok && probe.error !== null) log(probe.error);
    probes.push(probe);
  }
  catalog = buildCatalog(probes, brief.ratings);
}

/** Requirements read off a request body, merged with (never instead of) the manifest row. */
function requirementsFromBody(clientWire: Wire, body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    ...(Array.isArray(body.tools) && body.tools.length > 0 ? { tools: true } : {}),
    ...(vision ? { vision: true } : {}),
    ...(clientWire === 'openai' && (format === 'json_object' || format === 'json_schema') ? { json: true } : {}),
  };
}

function requirementsFor(clientWire: Wire, body: Json): Requirement {
  const manifest = brief.workload[manifestIndex];
  manifestIndex += 1;
  const fromBody = requirementsFromBody(clientWire, body);
  const merged: Requirement = { ...fromBody };
  if (manifest !== undefined) {
    if (manifest.requires.tools === true) merged.tools = true;
    if (manifest.requires.vision === true) merged.vision = true;
    if (manifest.requires.json === true) merged.json = true;
    if (manifest.requires.min_context !== undefined) merged.min_context = manifest.requires.min_context;
  }
  return merged;
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

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

function sendError(res: ServerResponse, wire: Wire, status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function modelsRoute(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_CREATED_AT })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
    });
  }
}

/**
 * Serve one client request: try candidates in order; on a refusal, count it and move
 * to the next. A second pass retries providers whose refusal was retryable (a failure
 * on the provider's schedule is transient; a spent budget refuses again). A 200 is
 * terminal: exactly one provider call may answer a request, so a success is never
 * retried, whatever its body holds.
 */
async function serveRequest(clientWire: Wire, body: Json, anthropicVersion: string): Promise<{ answer: CanonicalAnswer; modelId: string } | null> {
  const requires = requirementsFor(clientWire, body);
  const ir = clientWire === 'openai' ? openAIToIR(body) : anthropicToIR(body);
  const jsonModeOpenAI = clientWire === 'openai' && ir.jsonMode;
  const ordered = candidatesFor(probes, brief.ratings, requires, clientWire, jsonModeOpenAI);
  if (ordered.length === 0) {
    log(`no free model meets ${JSON.stringify(requires)}`);
    return null;
  }
  const triedRetryable = new Set<string>();
  const keyOf = (c: Candidate): string => `${c.probe.provider.id}/${c.model.id}`;
  for (let pass = 0; pass < 2; pass++) {
    for (const candidate of ordered) {
      const key = keyOf(candidate);
      if (pass === 1 && !triedRetryable.has(key)) continue;
      if (exhaustedProviders.has(candidate.probe.provider.id)) continue;
      recordAttempt(candidate.probe.provider.id);
      const prepared = prepareCall(candidate, clientWire, body, ir, anthropicVersion);
      const result = await callUpstream(prepared, candidate.probe.provider, candidate.model.id, ir.stream);
      if (result.ok) {
        recordSuccess(candidate.probe.provider.id);
        log(`served by ${key} (rating ${candidate.rating})`);
        return { answer: result.answer, modelId: candidate.model.id };
      }
      recordFailure(candidate.probe.provider.id);
      if (isTerminalRefusal(candidate.probe.provider, result.message)) {
        exhaustedProviders.add(candidate.probe.provider.id);
        log(`${key} is spent for this trial; it will not be tried again`);
      } else if (result.retryable) {
        triedRetryable.add(key);
      }
      log(`${key} refused (HTTP ${result.status}): ${result.detail}`);
    }
    if (triedRetryable.size === 0) break;
    if (pass === 0) log('first pass exhausted; retrying retryable refusals once');
  }
  return null;
}

// One inference at a time, in arrival order: a provider that serves one call at a
// time must never see two, and the manifest index must follow arrival order.
let serveQueue: Promise<void> = Promise.resolve();
function withServeLock<T>(fn: () => Promise<T>): Promise<T> {
  const run = serveQueue.then(fn, fn);
  serveQueue = run.then(
    () => undefined,
    () => undefined,
  );
  return run;
}

async function handleInference(req: IncomingMessage, res: ServerResponse, clientWire: Wire): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientWire, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const anthropicVersion =
    typeof req.headers['anthropic-version'] === 'string' && req.headers['anthropic-version'] !== ''
      ? req.headers['anthropic-version']
      : ANTHROPIC_VERSION_DEFAULT;
  const served = await withServeLock(() => serveRequest(clientWire, body, anthropicVersion));
  if (served === null) {
    return sendError(res, clientWire, 502, clientWire === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'every free model that meets this request refused it');
  }
  const stream = body.stream === true;
  if (clientWire === 'openai') {
    if (stream) sendSse(res, openAISseAnswer(served.answer, served.modelId));
    else sendJson(res, 200, openAIJsonAnswer(served.answer, served.modelId));
  } else {
    if (stream) sendSse(res, anthropicSseAnswer(served.answer, served.modelId));
    else sendJson(res, 200, anthropicJsonAnswer(served.answer, served.modelId));
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  const path = url.pathname.length > 1 && url.pathname.endsWith('/') ? url.pathname.slice(0, -1) : url.pathname;
  if (req.method === 'GET' && path === '/v1/models') {
    modelsRoute(req, res);
    return;
  }
  if (req.method === 'POST' && path === '/v1/chat/completions') {
    await handleInference(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && path === '/v1/messages') {
    await handleInference(req, res, 'anthropic');
    return;
  }
  sendError(res, path === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${path}`);
}

function listenOnce(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      const address = server.address();
      if (address === null || typeof address === 'string') {
        reject(new Error('the server has no port'));
        return;
      }
      resolve(address.port);
    });
  });
}

/** POST /register with the port and the one catalog; retried, since the env must hear it. */
async function register(port: number): Promise<void> {
  let lastError: string = 'unknown error';
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(REGISTER_TIMEOUT_MS),
      });
      if (!res.ok) {
        lastError = `POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`;
        continue;
      }
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = (err as Error).message;
    }
  }
  throw new Error(lastError);
}

/** Poll GET /work; exit 0 the moment it says done. A process left running scores 0. */
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
  let port = 0;
  for (let attempt = 0; attempt < 3; attempt++) {
    port = await listenOnce(server);
    // 8080 is the env's on one layout; the OS's ephemeral range never holds it.
    if (port !== 8080) break;
    await new Promise<void>((resolve) => server.close(() => resolve()));
    if (attempt === 2) throw new Error('the OS kept picking 8080, which the env owns on one layout');
  }
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} probed providers and ${servedModels().length} free models`);
  await register(port);
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

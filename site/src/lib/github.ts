import { REPO_NAME, REPO_OWNER } from "./site";

export type Difficulty = "good-first" | "intermediate" | "research" | "unlabeled";
export type Thrust = "foundations" | "mechanisms" | "robustness" | "theory";
export type Track = "A" | "B";

export interface IssueFacets {
  type?: "research-question" | "experiment" | "infrastructure";
  track?: Track;
  thrust?: Thrust;
  difficulty: Difficulty;
}

export interface Issue {
  number: number;
  title: string;
  url: string;
  labels: string[];
  facets: IssueFacets;
  comments: number;
  updatedAt: string;
  isDraft: boolean;
}

const API = `https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/issues?state=open&per_page=100`;
const CACHE_KEY = "ta-issues-v1";
const CACHE_TTL_MS = 10 * 60 * 1000;

export function deriveFacets(labels: string[]): IssueFacets {
  const has = (name: string) => labels.includes(name);
  const valueAfter = (prefix: string) =>
    labels.find((l) => l.startsWith(prefix))?.slice(prefix.length);

  let difficulty: Difficulty = "unlabeled";
  if (has("good first issue")) difficulty = "good-first";
  else if (has("difficulty:intermediate")) difficulty = "intermediate";
  else if (has("difficulty:research")) difficulty = "research";

  const trackRaw = valueAfter("track:");
  const track: Track | undefined = trackRaw?.startsWith("A")
    ? "A"
    : trackRaw?.startsWith("B")
      ? "B"
      : undefined;

  const thrustRaw = valueAfter("thrust:");
  const thrust = (["foundations", "mechanisms", "robustness", "theory"] as const).find(
    (t) => t === thrustRaw,
  );

  const type = (["research-question", "experiment", "infrastructure"] as const).find((t) =>
    has(t),
  );

  return { type, track, thrust, difficulty };
}

interface RawLabel {
  name: string;
}
interface RawIssue {
  number: number;
  title: string;
  html_url: string;
  comments: number;
  updated_at: string;
  pull_request?: unknown;
  labels: Array<RawLabel | string>;
}

function normalize(raw: RawIssue[]): Issue[] {
  return raw
    .filter((it) => !it.pull_request)
    .map((it) => {
      const labels = (it.labels ?? []).map((l) => (typeof l === "string" ? l : l.name));
      return {
        number: it.number,
        title: it.title.replace(/^DRAFT:\s*/i, "").trim(),
        url: it.html_url,
        labels,
        facets: deriveFacets(labels),
        comments: it.comments,
        updatedAt: it.updated_at,
        isDraft: /^DRAFT:/i.test(it.title) || labels.includes("draft"),
      };
    })
    .sort((a, b) => a.number - b.number);
}

function readCache(): Issue[] | null {
  if (typeof sessionStorage === "undefined") return null;
  try {
    const hit = sessionStorage.getItem(CACHE_KEY);
    if (!hit) return null;
    const { at, issues } = JSON.parse(hit);
    if (Date.now() - at > CACHE_TTL_MS) return null;
    return issues as Issue[];
  } catch {
    return null;
  }
}

function writeCache(issues: Issue[]): void {
  if (typeof sessionStorage === "undefined") return;
  try {
    sessionStorage.setItem(CACHE_KEY, JSON.stringify({ at: Date.now(), issues }));
  } catch {
    /* storage full or unavailable — ignore */
  }
}

export async function fetchIssues(signal?: AbortSignal): Promise<Issue[]> {
  const cached = readCache();
  if (cached) return cached;

  const res = await fetch(API, {
    signal,
    headers: { Accept: "application/vnd.github+json" },
  });
  if (!res.ok) {
    throw new Error(
      res.status === 403
        ? "GitHub rate limit reached. Please try again in a few minutes."
        : `GitHub API error (${res.status}).`,
    );
  }
  const issues = normalize((await res.json()) as RawIssue[]);
  writeCache(issues);
  return issues;
}

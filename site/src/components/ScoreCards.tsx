import { useEffect, useState } from "react";
import { BASE, RESULTS_URL, asset } from "../lib/site";

interface Claim {
  id: number;
  claim: string;
  metric: string;
  value: string;
  status: "verified" | "preliminary";
  source: string;
}
interface LayerRow {
  layer: number;
  train_accuracy: number | null;
  test_accuracy: number | null;
}
interface Separability {
  layers: LayerRow[];
  peak_train: { layer: number; accuracy: number };
  peak_test: { layer: number; accuracy: number };
}
interface Figure {
  name: string;
  file: string;
  caption: string;
}

interface SiteData {
  claims: Claim[];
  separability: Separability;
  figures: Figure[];
}

async function loadData(signal: AbortSignal): Promise<SiteData> {
  const get = async (name: string) => {
    const res = await fetch(`${BASE}data/${name}.json`, { signal });
    if (!res.ok) throw new Error(`Missing ${name}.json`);
    return res.json();
  };
  const [claims, separability, figures] = await Promise.all([
    get("claims"),
    get("probe_separability"),
    get("figures"),
  ]);
  return { claims: claims.claims, separability, figures: figures.figures };
}

function ProbeChart({ sep }: { sep: Separability }) {
  const W = 560;
  const H = 240;
  const padL = 40;
  const padR = 16;
  const padT = 16;
  const padB = 28;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const maxLayer = sep.layers[sep.layers.length - 1].layer;
  const yMin = 0.5;
  const yMax = 1.0;
  const x = (layer: number) => padL + (layer / maxLayer) * plotW;
  const y = (acc: number) => padT + (1 - (acc - yMin) / (yMax - yMin)) * plotH;
  const line = (key: "train_accuracy" | "test_accuracy") =>
    sep.layers
      .filter((r) => r[key] != null)
      .map((r) => `${x(r.layer)},${y(r[key] as number)}`)
      .join(" ");

  const ticks = [0.5, 0.75, 1.0];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Probe accuracy by layer">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={padL} x2={W - padR} y1={y(t)} y2={y(t)} className="stroke-rule" strokeWidth={1} />
          <text x={4} y={y(t) + 4} className="fill-ink-soft" fontSize={11}>
            {Math.round(t * 100)}%
          </text>
        </g>
      ))}
      <polyline points={line("train_accuracy")} fill="none" stroke="#059669" strokeWidth={2.5} />
      <polyline points={line("test_accuracy")} fill="none" stroke="#78716c" strokeWidth={2.5} strokeDasharray="4 3" />
      <circle cx={x(sep.peak_train.layer)} cy={y(sep.peak_train.accuracy)} r={4} fill="#059669" />
      <text x={x(sep.peak_train.layer)} y={y(sep.peak_train.accuracy) - 8} className="fill-accent-600" fontSize={11} textAnchor="middle">
        Layer {sep.peak_train.layer}: {Math.round(sep.peak_train.accuracy * 100)}%
      </text>
      {sep.layers.map((r) => (
        <text key={r.layer} x={x(r.layer)} y={H - 8} fill="#a8a29e" fontSize={10} textAnchor="middle">
          {r.layer}
        </text>
      ))}
      <text x={padL} y={H - 8} fill="#a8a29e" fontSize={10} textAnchor="middle" />
    </svg>
  );
}

export default function ScoreCards() {
  const [data, setData] = useState<SiteData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    loadData(ctrl.signal)
      .then(setData)
      .catch((e) => {
        if (e.name !== "AbortError") setError(String(e.message ?? e));
      });
    return () => ctrl.abort();
  }, []);

  if (error) {
    return (
      <div className="rounded-lg border border-amber-300 bg-amber-50 p-5 text-sm text-amber-800 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-300">
        Couldn't load the scored results.{" "}
        <a className="font-medium text-accent-700 underline dark:text-accent-400" href={RESULTS_URL}>
          Browse them on GitHub
        </a>
        . (Run <code className="rounded bg-stone-100 px-1 font-sans text-accent-800 dark:bg-stone-800 dark:text-accent-300">npm run data</code> to generate them locally.)
      </div>
    );
  }
  if (!data) {
    return <div className="animate-pulse font-sans text-ink-soft">Loading verified results…</div>;
  }

  return (
    <div className="space-y-10">
      <div className="grid gap-4 sm:grid-cols-2">
        {data.claims.map((c) => (
          <div key={c.id} className="rounded-lg border border-rule bg-surface p-5">
            <div className="flex items-start justify-between gap-3">
              <div className="font-display text-3xl font-bold text-ink">{c.value}</div>
              <span
                className={`rounded-full px-2 py-0.5 font-sans text-xs font-semibold uppercase tracking-wide ${
                  c.status === "verified"
                    ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300"
                    : "bg-amber-50 text-amber-700 dark:bg-amber-500/15 dark:text-amber-300"
                }`}
              >
                {c.status}
              </span>
            </div>
            <div className="mt-2 font-semibold text-ink">{c.claim}</div>
            <div className="mt-1 text-sm text-ink-soft">{c.metric}</div>
            <div className="mt-3 font-mono text-xs text-stone-500">{c.source}</div>
          </div>
        ))}
      </div>

      <div className="rounded-lg border border-rule bg-surface p-5">
        <h3 className="font-display text-lg font-bold text-ink">Probe accuracy by layer</h3>
        <p className="mt-1 font-sans text-sm text-ink-soft">
          Time-horizon is readable from a single layer and peaks mid-network.{" "}
          <span className="text-accent-700">train (solid)</span>,{" "}
          <span className="text-stone-500">test (dashed)</span>
        </p>
        <div className="mt-3">
          <ProbeChart sep={data.separability} />
        </div>
      </div>

      {data.figures.length > 0 && (
        <div className="grid gap-6">
          {data.figures.map((f) => (
            <figure key={f.name} className="rounded-lg border border-rule bg-surface p-4">
              <img
                src={asset(`figures/${f.file}`)}
                alt={f.caption}
                loading="lazy"
                className="w-full rounded bg-white"
              />
              <figcaption className="mt-2 font-sans text-sm text-ink-soft">{f.caption}</figcaption>
            </figure>
          ))}
        </div>
      )}
    </div>
  );
}

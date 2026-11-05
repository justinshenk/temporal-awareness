import { useEffect, useRef, useState } from "react";
import { GEOAPP_URL, REPO_URL } from "../lib/site";

function Fallback({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-rule bg-surface p-8 text-center">
      <div className="mb-2 text-4xl">🧭</div>
      <h3 className="font-display text-lg font-semibold text-ink">{title}</h3>
      <div className="mx-auto mt-2 max-w-md text-sm text-ink-soft">{children}</div>
    </div>
  );
}

export default function ExploreEmbed() {
  const [status, setStatus] = useState<"loading" | "ready" | "timeout">("loading");
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => {
    if (!GEOAPP_URL) return;
    timer.current = window.setTimeout(() => {
      setStatus((s) => (s === "ready" ? s : "timeout"));
    }, 12000);
    return () => window.clearTimeout(timer.current);
  }, []);

  if (!GEOAPP_URL) {
    return (
      <Fallback title="The interactive explorer runs against a live backend">
        <p>
          The geometry explorer (PCA scatter, layer trajectories, alignment heatmaps) is served by the
          project's FastAPI app. It isn't wired to a public host yet.
        </p>
        <p className="mt-3">
          Run it locally from{" "}
          <a className="font-medium text-accent-700 underline" href={`${REPO_URL}/tree/main/src/intertemporal/geoapp`}>
            <code>src/intertemporal/geoapp</code>
          </a>
          , or set <code className="rounded bg-stone-100 px-1 font-sans text-accent-800 dark:bg-stone-800 dark:text-accent-300">PUBLIC_GEOAPP_URL</code> at build time to embed it here.
        </p>
      </Fallback>
    );
  }

  return (
    <div>
      {status === "timeout" && (
        <div className="mb-3 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-800 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-300">
          The explorer is taking a while; the backend may be waking up.{" "}
          <a className="font-medium text-accent-700 underline" href={GEOAPP_URL} target="_blank" rel="noreferrer">
            Open it in a new tab
          </a>
          .
        </div>
      )}
      <div className="h-[88svh] min-h-[640px] w-full overflow-hidden rounded-lg border border-rule bg-surface lg:h-auto lg:min-h-0 lg:aspect-[16/10]">
        <iframe
          src={GEOAPP_URL}
          title="Temporal geometry explorer"
          className="h-full w-full"
          onLoad={() => setStatus("ready")}
        />
      </div>
    </div>
  );
}

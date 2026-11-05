// Guard against the base-path concatenation bug.
//
// On a GitHub Pages *project* deploy the site is served under a base path
// (e.g. "/temporal-awareness"). Astro's `import.meta.env.BASE_URL` drops the
// trailing slash when `base` has none, so a naive `${BASE}understand` join
// produces "/temporal-awarenessunderstand" instead of
// "/temporal-awareness/understand" — 404-ing every internal link. This worked
// locally (base "/") which is exactly why it slipped through.
//
// Run after `astro build` with the same PUBLIC_BASE_PATH the build used:
//   PUBLIC_BASE_PATH=/temporal-awareness node scripts/site/check_built_links.mjs
//
// Exits non-zero if any built page contains an internal link whose path segment
// is glued directly onto the base path (the missing-slash bug).

import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";

const DIST = fileURLToPath(new URL("../../site/dist/", import.meta.url));

// Normalize the base to a leading slash with no trailing slash, e.g. "/temporal-awareness".
let base = process.env.PUBLIC_BASE_PATH ?? "";
base = base.replace(/\/+$/g, "");
if (base && !base.startsWith("/")) base = `/${base}`;

if (!base) {
  console.log("PUBLIC_BASE_PATH is empty (root deploy): no base path to glue, nothing to check.");
  process.exit(0);
}

function collectHtml(dir) {
  const files = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) files.push(...collectHtml(full));
    else if (entry.endsWith(".html")) files.push(full);
  }
  return files;
}

const escaped = base.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
// A correct internal link reads `="${base}/..."`. The bug glues a path segment
// straight onto the base, so the character after the base is neither "/" nor the
// closing quote. That lookahead is the whole tell.
const glued = new RegExp(`(?:href|src)="${escaped}(?=[^/"])`, "g");

let htmlFiles;
try {
  htmlFiles = collectHtml(DIST);
} catch {
  console.error(`✗ No build found at ${DIST}. Run \`npm run build\` first.`);
  process.exit(1);
}

const offenders = [];
for (const file of htmlFiles) {
  const matches = readFileSync(file, "utf8").match(glued);
  if (matches) {
    const rel = file.slice(DIST.length);
    offenders.push(`  ${rel}: ${[...new Set(matches)].sort().join(", ")}`);
  }
}

if (offenders.length) {
  console.error(`✗ ${offenders.length} built page(s) have internal links glued onto the base path (missing slash):`);
  console.error(offenders.join("\n"));
  console.error(`\nThe page would 404 on ${base}/. Fix: ensure BASE ends with a trailing slash in site/src/lib/site.ts.`);
  process.exit(1);
}

// A correctly prefixed link can still point at a page that no longer exists.
const internalLink = new RegExp(`(?:href|src)="${escaped}/([^"#?]*)`, "g");
const missing = new Set();
for (const file of htmlFiles) {
  const rel = file.slice(DIST.length);
  const html = readFileSync(file, "utf8");
  // A redirect stub must stay on this deployment, not jump to another domain.
  if (/http-equiv="refresh"[^>]*url=https?:/.test(html)) missing.add(`  ${rel}: redirect leaves this deployment`);
  for (const [, target] of html.matchAll(internalLink)) {
    const candidates = [join(DIST, target), join(DIST, target, "index.html")];
    const resolves = candidates.some((path) => existsSync(path) && (statSync(path).isFile() || target === ""));
    if (!resolves) missing.add(`  ${rel}: ${base}/${target}`);
  }
}

if (missing.size) {
  console.error(`✗ ${missing.size} internal link(s) point at nothing in the build:`);
  console.error([...missing].sort().join("\n"));
  process.exit(1);
}

console.log(`✓ ${htmlFiles.length} built page(s): every internal link is prefixed with ${base}/ and resolves`);

export const REPO_OWNER = "justinshenk";
export const REPO_NAME = "temporal-awareness";
export const REPO_URL = `https://github.com/${REPO_OWNER}/${REPO_NAME}`;
export const ISSUES_URL = `${REPO_URL}/issues`;
export const GOOD_FIRST_ISSUES_URL = `${ISSUES_URL}?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22`;
export const CONTRIBUTING_URL = `${REPO_URL}/blob/main/docs/CONTRIBUTING.md`;
export const RESEARCH_PROGRAM_URL = `${REPO_URL}/blob/main/docs/RESEARCH_PROGRAM.md`;

/** Hosted geoapp FastAPI URL. Set PUBLIC_GEOAPP_URL at build time to enable the live explorer. */
export const GEOAPP_URL: string = import.meta.env.PUBLIC_GEOAPP_URL ?? "";

/** Prefix for static assets / generated data, respecting the deploy base path.
 * Astro's BASE_URL omits the trailing slash when `base` has none (e.g. the GitHub
 * Pages project path "/temporal-awareness"), which would turn `${BASE}understand`
 * into "/temporal-awarenessunderstand" and 404 every internal link. Always end with
 * exactly one slash so each `${BASE}path` join is well-formed. */
const rawBase: string = import.meta.env.BASE_URL;
export const BASE: string = rawBase.endsWith("/") ? rawBase : `${rawBase}/`;
export const asset = (path: string): string => `${BASE}${path.replace(/^\//, "")}`;

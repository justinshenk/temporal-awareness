import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import tailwind from "@astrojs/tailwind";
import mdx from "@astrojs/mdx";

// For a GitHub Pages project site, the deploy workflow sets PUBLIC_BASE_PATH=/temporal-awareness.
// Locally and at the domain root it defaults to "/".
const base = (process.env.PUBLIC_BASE_PATH || "").replace(/\/+$/, "");

// No `site` here: the build deploys to two domains, and Astro would prefix redirect targets with it.
export default defineConfig({
  base: base || "/",
  integrations: [react(), tailwind(), mdx()],
  redirects: {
    "/understand": `${base}/representation`,
    "/what-we-score": `${base}/evidence`,
    "/start-here": `${base}/contribute`,
  },
});

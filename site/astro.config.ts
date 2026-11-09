import { defineConfig } from "astro/config";
import tailwindcss from "@tailwindcss/vite";
import icon from "astro-icon";
import mdx from "@astrojs/mdx";

export default defineConfig({
  output: "static",
  site: process.env.SITE_URL || "https://aswahd.github.io",
  base: process.env.BASE_PATH || "/",
  vite: { plugins: [tailwindcss()] },
  integrations: [icon(), mdx()],
});

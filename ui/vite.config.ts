import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
// @ts-expect-error type error without @types/node package
import process from "node:process";
// @ts-expect-error type error without @types/node package
import { createHash } from "node:crypto";
// @ts-expect-error type error without @types/node package
import { readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
// @ts-expect-error type error without @types/node package
import { join, relative } from "node:path";
const host = process.env.TAURI_DEV_HOST;

/** dist/.build-id: a hash of the built page. NEO.app copies its page (Resources/ui) to Application
 *  Support/NEO/ui for NotchHub whenever the ids differ (ui/src-tauri/src/setup.rs `sync_ui`). */
function buildId(): Plugin {
  let outDir = "dist";
  return {
    name: "neo-build-id",
    apply: "build",
    configResolved(c) { outDir = c.build.outDir; },
    closeBundle() {
      const files: string[] = [];
      const walk = (d: string) => {
        for (const n of readdirSync(d).sort()) {
          const p = join(d, n);
          if (statSync(p).isDirectory()) walk(p);
          else if (n !== ".build-id") files.push(p);
        }
      };
      walk(outDir);
      const h = createHash("sha256");
      for (const f of files) h.update(`${relative(outDir, f)}\n`).update(readFileSync(f));
      writeFileSync(join(outDir, ".build-id"), `${h.digest("hex").slice(0, 16)}\n`);
    },
  };
}

// https://vite.dev/config/
export default defineConfig(() => ({
  plugins: [react(), buildId()],

  // Vite options tailored for Tauri development and only applied in `tauri dev` or `tauri build`
  //
  // 1. prevent Vite from obscuring rust errors
  clearScreen: false,
  // 2. tauri expects a fixed port, fail if that port is not available
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? {
          protocol: "ws",
          host,
          port: 1421,
        }
      : undefined,
    watch: {
      // 3. tell Vite to ignore watching `src-tauri`
      ignored: ["**/src-tauri/**"],
    },
  },
}));

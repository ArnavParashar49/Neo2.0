import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { PopoutApp } from "./browser/PopoutApp";
import { CameraPopoutApp } from "./camera/CameraPopoutApp";
import { KeysApp } from "./keys/KeysApp";
import "./App.css";
import "./browser/browser.css";

// index.html#/browser is the pop-out window: just the browser card, full size.
// index.html#/camera is the camera's pop-out: just the camera card.
// index.html#/keys is Brains & Keys: NEO.app's shell alone opens it (ui/src-tauri/src/keys.rs).
const popout = location.hash === "#/browser";
const camera = location.hash === "#/camera";
const keys = location.hash === "#/keys";
if (popout || camera) document.documentElement.dataset.surface = "window";
if (keys) {
  // A settings window: it follows the Mac's appearance (the overlay ships dark).
  document.documentElement.dataset.surface = "keys";
  const light = typeof matchMedia === "function" ? matchMedia("(prefers-color-scheme: light)") : null;
  const follow = () => { document.documentElement.dataset.theme = light?.matches ? "light" : "dark"; };
  follow();
  light?.addEventListener?.("change", follow);
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    {keys ? <KeysApp /> : camera ? <CameraPopoutApp /> : popout ? <PopoutApp /> : <App />}
  </React.StrictMode>,
);

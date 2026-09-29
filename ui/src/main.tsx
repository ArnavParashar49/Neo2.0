import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { PopoutApp } from "./browser/PopoutApp";
import { CameraPopoutApp } from "./camera/CameraPopoutApp";
import "./App.css";
import "./browser/browser.css";

// index.html#/browser is the pop-out window: just the browser card, full size.
// index.html#/camera is the camera's pop-out: just the camera card.
const popout = location.hash === "#/browser";
const camera = location.hash === "#/camera";
if (popout || camera) document.documentElement.dataset.surface = "window";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    {camera ? <CameraPopoutApp /> : popout ? <PopoutApp /> : <App />}
  </React.StrictMode>,
);

import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { PopoutApp } from "./browser/PopoutApp";
import "./App.css";
import "./browser/browser.css";

// index.html#/browser is the pop-out window: just the browser card, full size.
const popout = location.hash === "#/browser";
if (popout) document.documentElement.dataset.surface = "window";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    {popout ? <PopoutApp /> : <App />}
  </React.StrictMode>,
);

import React from "react";
import ReactDOM from "react-dom/client";
import { TerminalShell } from "./terminal/TerminalShell";
import "./styles.css";

const rootElement = document.getElementById("root") as HTMLElement;

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <TerminalShell />
  </React.StrictMode>
);


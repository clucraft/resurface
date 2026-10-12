import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { AppShell } from "./ui/AppShell";
import "./ui/styles.css";

if (import.meta.env.DEV) void import("./dev/testkit");

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <AppShell />
  </StrictMode>,
);

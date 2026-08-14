import { invoke } from "@tauri-apps/api/core";
import { writeText } from "@tauri-apps/plugin-clipboard-manager";
import { fetch as tauriFetch } from "@tauri-apps/plugin-http";
import { openUrl } from "@tauri-apps/plugin-opener";
import { Check, Clipboard, ExternalLink, Loader2, Server, Terminal } from "lucide-react";
import { useState } from "react";
import { RunnerInfo } from "../../lib/types";
import { cn } from "../../lib/classNames";
import { Button, Card, ErrorBlock } from "../ui";

export function AgentsPanel() {
  const [error, setError] = useState("");
  const [apiUrl, setApiUrl] = useState("");
  const [startingApi, setStartingApi] = useState(false);
  const [copied, setCopied] = useState<"agent" | "curl" | "">("");

  async function startApi() {
    setStartingApi(true);
    setError("");
    try {
      const info = await invoke<RunnerInfo>("start_runner");
      setApiUrl(info.base_url);
      return info.base_url;
    } catch (err) {
      setError(String(err));
      return "";
    } finally {
      setStartingApi(false);
    }
  }

  async function openApiDocs() {
    const baseUrl = apiUrl || (await startApi());
    if (baseUrl) await openUrl(`${baseUrl}/docs`);
  }

  async function copyText(kind: "agent" | "curl", text: string) {
    await writeText(text);
    setCopied(kind);
    window.setTimeout(() => setCopied(""), 1600);
  }

  async function copyAgentSkill() {
    const baseUrl = apiUrl || (await startApi());
    if (!baseUrl) return;
    try {
      const response = await tauriFetch(`${baseUrl}/skill`);
      if (!response.ok) throw new Error(`failed to fetch skill (${response.status})`);
      await copyText("agent", await response.text());
    } catch (err) {
      setError(String(err));
    }
  }

  const shownApiUrl = apiUrl || "Start the local API to see the URL";
  const curlExamples = apiUrl
    ? `curl ${apiUrl}/health

curl ${apiUrl}/openapi.json

curl -X POST ${apiUrl}/v1/models/load \\
  -H 'Content-Type: application/json' \\
  -d '{}'

curl -X POST ${apiUrl}/v1/audio/speech \\
  -H 'Content-Type: application/json' \\
  -o speech.wav \\
  -d '{"input":"Hello from MamboRambo","language":"auto","response_format":"wav"}'`
    : `curl http://127.0.0.1:<port>/health
curl http://127.0.0.1:<port>/openapi.json`;

  return (
    <div className="space-y-4">
      <Card className="divide-y divide-border/20 overflow-hidden border-none shadow-xl">
        <div className="space-y-4 p-6">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="min-w-0 space-y-1">
              <p className="flex items-center gap-2 text-base font-semibold tracking-tight text-primary">
                <Server className="h-4 w-4 text-secondary opacity-40" />
                MamboRambo HTTP API
              </p>
              <p className="text-xs leading-5 text-secondary opacity-50">Swagger docs, OpenAPI schema, and agent-ready examples.</p>
            </div>
            <Button
              variant={apiUrl ? "secondary" : "primary"}
              onClick={startApi}
              disabled={startingApi}
              className="h-8 shrink-0 gap-2 rounded-full px-3 text-[10px] font-black uppercase tracking-[0.16em]"
            >
              {startingApi ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <span className={cn("h-1.5 w-1.5 rounded-full", apiUrl ? "bg-green-500" : "bg-white/70")} />
              )}
              {apiUrl ? "Running" : "Start API"}
            </Button>
          </div>

          <p className="truncate rounded-lg border border-border/10 bg-background/50 px-3 py-2 font-mono text-[11px] text-secondary/70">
            {shownApiUrl}
          </p>

          <div className="grid gap-2 sm:grid-cols-3">
            <Button variant="outline" onClick={openApiDocs} className="h-9 gap-2 px-3 text-[11px]">
              <ExternalLink className="h-4 w-4" />
              Swagger
            </Button>
            <Button variant="secondary" onClick={copyAgentSkill} className="h-9 gap-2 px-3 text-[11px]">
              {copied === "agent" ? <Check className="h-4 w-4" /> : <Clipboard className="h-4 w-4" />}
              {copied === "agent" ? "Copied" : "Agent Skill"}
            </Button>
            <Button variant="secondary" onClick={() => copyText("curl", curlExamples)} className="h-9 gap-2 px-3 text-[11px]">
              {copied === "curl" ? <Check className="h-4 w-4" /> : <Terminal className="h-4 w-4" />}
              {copied === "curl" ? "Copied" : "cURL"}
            </Button>
          </div>
        </div>
      </Card>

      {error && <ErrorBlock>{error}</ErrorBlock>}
    </div>
  );
}

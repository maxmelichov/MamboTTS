import { Bot } from "lucide-react";
import type { ModelBundle } from "../lib/types";
import { AgentsPanel } from "../components/agents/AgentsPanel";
import { AppFrame } from "../components/AppFrame";
import { WorkspaceHeader } from "../components/WorkspaceHeader";

export function AgentsPage({ bundle }: { bundle: ModelBundle | null }) {
  return (
    <AppFrame bundle={bundle}>
      <section className="w-full max-w-[1200px]">
        <WorkspaceHeader bundle={bundle} active="api" />

        <div className="mx-auto mt-4 max-w-[760px] space-y-10">
          <header className="space-y-3">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-border/40 bg-white shadow-sm">
                <Bot className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-[10px] font-black uppercase tracking-[0.24em] text-secondary opacity-35">Local Service</p>
                <h1 className="text-3xl font-semibold tracking-tight text-primary sm:text-4xl">API Workspace</h1>
              </div>
            </div>
            <p className="max-w-[520px] text-base leading-7 text-secondary opacity-60">
              Start MamboRambo's local HTTP API, open Swagger, and copy agent-ready instructions for using speech synthesis from tools.
            </p>
          </header>

          <AgentsPanel />
        </div>
      </section>
    </AppFrame>
  );
}

import { getVersion } from "@tauri-apps/api/app";
import { AudioLines, Bot, Settings } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import type { ModelBundle } from "../lib/types";
import { cn } from "../lib/classNames";
import { Brand } from "./ui";

type Workspace = "studio" | "api";

export function StudioHeader({ bundle }: { bundle: ModelBundle | null }) {
  return <WorkspaceHeader bundle={bundle} active="studio" />;
}

export function WorkspaceHeader({ bundle, active }: { bundle: ModelBundle | null; active: Workspace }) {
  const [appVersion, setAppVersion] = useState("");

  useEffect(() => {
    getVersion().then(setAppVersion).catch(() => undefined);
  }, []);

  return (
    <header className="flex h-[88px] items-center justify-between">
      <div className="flex items-center gap-6">
        <Brand />
        <MainNav active={active} />
      </div>
      <div className="flex items-center gap-4">
        <span className="text-[10px] font-medium tracking-widest text-secondary opacity-40 uppercase">
          {bundle?.version.split("-").pop() ?? appVersion}
        </span>
        <Link to="/settings" className="flex h-8 w-8 items-center justify-center rounded-full border border-border/60 text-secondary transition-all hover:border-primary hover:text-primary">
          <Settings className="h-4 w-4" />
        </Link>
      </div>
    </header>
  );
}

function MainNav({ active }: { active: Workspace }) {
  const items = [
    { id: "studio", label: "Studio", to: "/home", icon: AudioLines },
    { id: "api", label: "API", to: "/agents", icon: Bot },
  ] as const;

  return (
    <nav className="inline-flex min-w-[220px] rounded-2xl border border-border/30 bg-white p-1.5 shadow-sm">
      {items.map((item) => {
        const Icon = item.icon;
        const isActive = active === item.id;
        return (
          <Link
            key={item.id}
            to={item.to}
            className={cn(
              "flex h-10 flex-1 items-center justify-center gap-2 rounded-xl px-4 text-[11px] font-black uppercase tracking-[0.16em] transition-all",
              isActive ? "bg-primary text-white shadow-sm" : "text-secondary opacity-55 hover:text-primary hover:opacity-100",
            )}
          >
            <Icon className="h-4 w-4" />
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}

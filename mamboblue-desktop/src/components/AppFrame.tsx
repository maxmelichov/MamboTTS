import { ReactNode } from "react";
import { ModelBundle } from "../lib/types";

export function AppFrame({ children }: { bundle: ModelBundle | null; children: ReactNode }) {
  return (
    <main className="min-h-screen bg-background p-6 text-primary selection:bg-primary selection:text-white sm:p-12">
      <div className="mx-auto flex w-full justify-center">{children}</div>
    </main>
  );
}

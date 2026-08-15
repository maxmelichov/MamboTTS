import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";
import { CreateStep } from "../lib/types";
import { Card } from "./ui";

export function CreateStatus({ step, status }: { step: CreateStep; status: string }) {
  const steps: Array<[CreateStep, string]> = [
    ["starting", "Initializing"],
    ["loading", "Loading Models"],
    ["creating", "Generating"],
  ];
  const activeIndex = steps.findIndex(([key]) => key === step);

  return (
    <Card className="border-none bg-surface shadow-lg rounded-xl">
      <div className="flex items-center gap-3 p-4">
        <Loader2 className="h-4 w-4 animate-spin text-primary" />
        <div className="flex-1">
          <p className="text-sm font-semibold">{status}</p>
          <div className="mt-2 h-1 w-full overflow-hidden rounded-full bg-border/40">
            <motion.div
              className="h-full bg-primary"
              initial={{ width: "0%" }}
              animate={{ width: `${((activeIndex + 1) / steps.length) * 100}%` }}
              transition={{ type: "spring", bounce: 0, duration: 0.5 }}
            />
          </div>
        </div>
      </div>
    </Card>
  );
}

import { motion } from "framer-motion";
import { X } from "lucide-react";
import { ComponentProps, ReactNode } from "react";
import mamboramboLogo from "../assets/mamborambo-mark.png";
import { cn } from "../lib/classNames";

type ButtonVariant = "primary" | "secondary" | "ghost" | "outline";

export function Button({ className, variant = "primary", ...props }: ComponentProps<"button"> & { variant?: ButtonVariant }) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center rounded-lg text-sm font-semibold cursor-pointer transition-all active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-50",
        variant === "primary" && "bg-[#111111] text-white hover:bg-black shadow-sm shadow-primary/10",
        variant === "secondary" && "bg-white border border-border/60 text-primary hover:bg-background hover:border-border shadow-sm",
        variant === "outline" && "border border-border/80 bg-white text-primary hover:border-primary hover:bg-white shadow-sm",
        variant === "ghost" && "bg-transparent text-secondary hover:bg-background hover:text-primary",
        className,
      )}
      {...props}
    />
  );
}

export function Card({ className, ...props }: ComponentProps<"div">) {
  return <div className={cn("rounded-2xl border border-border/50 bg-white shadow-[0_8px_30px_rgb(0,0,0,0.04)]", className)} {...props} />;
}

export function Progress({ value }: { value: number }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-border/20">
      <motion.div
        className="h-full bg-primary"
        initial={false}
        animate={{ width: `${Math.max(0, Math.min(100, value))}%` }}
        transition={{ duration: 0.2, ease: "easeOut" }}
      />
    </div>
  );
}

export function Brand({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "inline-flex h-9 min-w-[72px] items-center justify-center gap-2.5 rounded-xl border border-border/40 bg-white px-3 text-[11px] font-black uppercase tracking-[0.25em] text-primary shadow-sm",
        className,
      )}
    >
      <img src={mamboramboLogo} alt="" className="h-5 w-5 shrink-0" />
      <span>MamboRambo</span>
    </div>
  );
}

export function LogoMark({ className }: { className?: string }) {
  return <img src={mamboramboLogo} alt="" className={cn("shrink-0", className)} />;
}

export function Eyebrow({ children, className }: { children: ReactNode; className?: string }) {
  return <span className={cn("mb-3 block text-[10px] font-bold uppercase tracking-[0.2em] text-secondary opacity-50", className)}>{children}</span>;
}

export function ErrorBlock({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn("flex gap-3 rounded-2xl border border-red-100 bg-red-50/30 p-5 text-sm text-red-900 shadow-sm", className)}>
      <X className="h-4 w-4 shrink-0 mt-0.5 opacity-60" />
      <div className="overflow-auto">
        <p className="font-semibold mb-1">System Error</p>
        <pre className="whitespace-pre-wrap break-words font-medium opacity-80">{children}</pre>
      </div>
    </div>
  );
}

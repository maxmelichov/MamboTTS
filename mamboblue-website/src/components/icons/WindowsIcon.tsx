import { cn } from "@/lib/utils"

export function WindowsIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 256 256" fill="currentColor" className={cn("size-5", className)} aria-hidden="true">
      <path d="M112 144v51.6a8 8 0 0 1-9.4 7.9l-64-11.6A8 8 0 0 1 32 184v-40a8 8 0 0 1 8-8h64a8 8 0 0 1 8 8Zm-2.9-89.8a8 8 0 0 0-6.5-1.7l-64 11.6A8 8 0 0 0 32 72v40a8 8 0 0 0 8 8h64a8 8 0 0 0 8-8V60.4a8 8 0 0 0-2.9-6.2ZM216 136h-80a8 8 0 0 0-8 8v57.5a8 8 0 0 0 6.6 7.9l80 14.5A8 8 0 0 0 224 216v-72a8 8 0 0 0-8-8Zm5.1-102.1a8 8 0 0 0-6.5-1.7l-80 14.5a8 8 0 0 0-6.6 7.9V112a8 8 0 0 0 8 8h80a8 8 0 0 0 8-8V40a8 8 0 0 0-2.9-6.1Z" />
    </svg>
  )
}

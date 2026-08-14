import { LinuxIcon } from "@/components/icons/LinuxIcon"
import { MacIcon } from "@/components/icons/MacIcon"
import { WindowsIcon } from "@/components/icons/WindowsIcon"
import type { Platform } from "@/lib/platform"

export function PlatformIcon({ platform, className }: { platform: Platform; className?: string }) {
  if (platform === "windows") return <WindowsIcon className={className} />
  if (platform === "linux") return <LinuxIcon className={className} />

  return <MacIcon className={className} />
}

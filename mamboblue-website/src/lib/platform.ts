export type Platform = "macos" | "windows" | "linux"

interface NavigatorWithUserAgentData extends Navigator {
  userAgentData?: {
    platform?: string
  }
}

export const platformLabels: Record<Platform, string> = {
  macos: "macOS",
  windows: "Windows",
  linux: "Linux",
}

export function detectPlatform(): Platform {
  const nav = navigator as NavigatorWithUserAgentData
  const platform =
    nav.userAgentData?.platform?.toLowerCase() ??
    navigator.platform?.toLowerCase() ??
    navigator.userAgent.toLowerCase()

  if (platform.includes("win")) return "windows"
  if (platform.includes("linux")) return "linux"

  return "macos"
}

export function isMobileDevice() {
  const userAgent = navigator.userAgent.toLowerCase()

  return /android|iphone|ipad|ipod|mobile/.test(userAgent) || (navigator.maxTouchPoints > 1 && window.screen.width < 768)
}

import { Monitor, Smartphone } from "lucide-react"
import { useMemo, useState } from "react"

import { GithubIcon } from "@/components/icons/GithubIcon"
import { PlatformIcon } from "@/components/icons/PlatformIcon"
import { LinuxInstallModal } from "@/components/LinuxInstallModal"
import { Button } from "@/components/ui/button"
import { githubUrl, releasesUrl } from "@/lib/links"
import latestReleaseJson from "@/lib/latest_release.json"
import type { LatestRelease, ReleaseAsset } from "@/lib/latestRelease"
import { detectPlatform, isMobileDevice, type Platform, platformLabels } from "@/lib/platform"
import { cn } from "@/lib/utils"

const latestRelease = latestReleaseJson as LatestRelease
const installCommand = `curl -sSf https://maxmelichov.github.io/MamboRambo/installer.sh | sh -s ${latestRelease.version}`

function preferredAsset(platform: Platform): ReleaseAsset | undefined {
  const assets = latestRelease.assets.filter((asset) => asset.platform === platform)

  if (platform === "windows") {
    return assets.find((asset) => asset.kind === "exe") ?? assets.find((asset) => asset.kind === "msi") ?? assets.find((asset) => asset.kind === "zip")
  }

  if (platform === "linux") {
    return assets.find((asset) => asset.kind === "appimage") ?? assets.find((asset) => asset.kind === "deb") ?? assets[0]
  }

  return assets.find((asset) => asset.arch === "darwin-aarch64") ?? assets[0]
}

export function DownloadCta() {
  const [platform, setPlatform] = useState<Platform>(() => detectPlatform())
  const [isMobile, setIsMobile] = useState(() => isMobileDevice())
  const [linuxModalOpen, setLinuxModalOpen] = useState(false)
  const asset = preferredAsset(platform)

  const downloadLabel = useMemo(() => {
    if (isMobile) return "Download on desktop"

    return `Download for ${platformLabels[platform]}`
  }, [isMobile, platform])

  return (
    <div className="flex flex-col items-center w-full">
      <div className="mt-4 flex flex-col items-center justify-center gap-4 sm:flex-row">
        {platform === "linux" && !isMobile ? (
          <Button size="lg" className="h-14 w-full sm:w-auto rounded-2xl px-8 text-base shadow-sm transition-all hover:scale-[1.02] active:scale-[0.98]" onClick={() => setLinuxModalOpen(true)}>
            <PlatformIcon platform={platform} className="size-5" />
            {downloadLabel}
          </Button>
        ) : (
          <Button size="lg" asChild className="h-14 w-full sm:w-auto rounded-2xl px-8 text-base shadow-sm transition-all hover:scale-[1.02] active:scale-[0.98]">
            <a href={isMobile ? releasesUrl : asset?.url ?? releasesUrl}>
              {isMobile ? <Smartphone className="size-5" /> : <PlatformIcon platform={platform} className="size-5" />}
              {downloadLabel}
            </a>
          </Button>
        )}
        <Button variant="outline" size="lg" asChild className="h-14 w-full sm:w-auto rounded-2xl px-8 text-base border-border/60 hover:bg-white hover:border-border transition-all hover:scale-[1.02] active:scale-[0.98]">
          <a href={githubUrl} target="_blank" rel="noreferrer">
            <GithubIcon className="size-5" />
            View on GitHub
          </a>
        </Button>
      </div>

      <div className="mt-8 flex items-center justify-center gap-1 rounded-[20px] border border-border/40 bg-white/50 p-1 backdrop-blur-sm shadow-sm max-w-fit mx-auto">
        {(Object.keys(platformLabels) as Platform[]).map((option) => (
          <button
            key={option}
            type="button"
            onClick={() => {
              setPlatform(option)
              setIsMobile(false)
            }}
            className={cn(
              "relative inline-flex h-9 px-4 items-center justify-center rounded-2xl text-[13px] font-semibold text-muted-foreground transition-all duration-300 hover:text-foreground",
              platform === option && !isMobile ? "bg-white text-foreground shadow-sm ring-1 ring-black/[0.03]" : "hover:bg-black/[0.02]"
            )}
            aria-label={`Select ${platformLabels[option]}`}
          >
            <PlatformIcon platform={option} className="mr-2 size-3.5" />
            {platformLabels[option]}
          </button>
        ))}
      </div>

      <div className="mt-6 flex items-center justify-center gap-2 text-[13px] font-medium text-muted-foreground/60">
        {isMobile ? <Smartphone className="size-3.5" /> : <Monitor className="size-3.5" />}
        <span>
          {isMobile
            ? "Available for macOS, Windows, and Linux."
            : `${latestRelease.version.includes('v') ? latestRelease.version : 'v' + latestRelease.version} for ${platformLabels[platform]}.`}
        </span>
      </div>

      <LinuxInstallModal command={installCommand} open={linuxModalOpen} onOpenChange={setLinuxModalOpen} />
    </div>
  )
}

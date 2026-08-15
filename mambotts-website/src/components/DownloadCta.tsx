import { Monitor, Smartphone } from "lucide-react"
import { useMemo, useState } from "react"

import { GithubIcon } from "@/components/icons/GithubIcon"
import { PlatformIcon } from "@/components/icons/PlatformIcon"
import { Button } from "@/components/ui/button"
import { githubUrl, releasesUrl } from "@/lib/links"
import latestReleaseJson from "@/lib/latest_release.json"
import type { LatestRelease, ReleaseAsset } from "@/lib/latestRelease"
import { detectPlatform, isMobileDevice, platformLabels } from "@/lib/platform"

const latestRelease = latestReleaseJson as LatestRelease

function macAsset(): ReleaseAsset | undefined {
  const assets = latestRelease.assets.filter((asset) => asset.platform === "macos")

  return assets.find((asset) => asset.arch === "darwin-aarch64") ?? assets[0]
}

export function DownloadCta() {
  const [isMobile] = useState(() => isMobileDevice())
  const [isMac] = useState(() => detectPlatform() === "macos")
  const asset = macAsset()

  const downloadLabel = useMemo(() => {
    if (isMobile) return "Download on desktop"

    return `Download for ${platformLabels.macos}`
  }, [isMobile])

  return (
    <div className="flex flex-col items-center w-full">
      <div className="mt-4 flex flex-col items-center justify-center gap-4 sm:flex-row">
        <Button size="lg" asChild className="h-14 w-full sm:w-auto rounded-2xl px-8 text-base shadow-sm transition-all hover:scale-[1.02] active:scale-[0.98]">
          <a href={isMobile ? releasesUrl : asset?.url ?? releasesUrl}>
            {isMobile ? <Smartphone className="size-5" /> : <PlatformIcon platform="macos" className="size-5" />}
            {downloadLabel}
          </a>
        </Button>
        <Button variant="outline" size="lg" asChild className="h-14 w-full sm:w-auto rounded-2xl px-8 text-base border-border/60 hover:bg-white hover:border-border transition-all hover:scale-[1.02] active:scale-[0.98]">
          <a href={githubUrl} target="_blank" rel="noreferrer">
            <GithubIcon className="size-5" />
            View on GitHub
          </a>
        </Button>
      </div>

      <div className="mt-6 flex items-center justify-center gap-2 text-[13px] font-medium text-muted-foreground/60">
        {isMobile ? <Smartphone className="size-3.5" /> : <Monitor className="size-3.5" />}
        <span>
          {isMobile || !isMac
            ? "Requires macOS on Apple Silicon."
            : `${latestRelease.version.includes("v") ? latestRelease.version : "v" + latestRelease.version} for ${platformLabels.macos}.`}
        </span>
      </div>
    </div>
  )
}

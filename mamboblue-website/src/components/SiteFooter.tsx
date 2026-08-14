import { githubUrl, issuesUrl, releasesUrl } from "@/lib/links"

export function SiteFooter() {
  return (
    <footer className="mx-auto w-full max-w-5xl px-4 py-12 text-sm text-muted-foreground/60 sm:px-6">
      <div className="flex flex-col items-center justify-between gap-6 border-t border-border/40 pt-12 sm:flex-row">
        <div className="flex items-center gap-2 font-bold tracking-tight text-foreground/40">
          <span>MamboRambo</span>
        </div>
        <nav className="flex items-center gap-8 font-medium">
          <a className="transition-colors hover:text-foreground" href={githubUrl} target="_blank" rel="noreferrer">
            GitHub
          </a>
          <a className="transition-colors hover:text-foreground" href={releasesUrl}>
            Releases
          </a>
          <a className="transition-colors hover:text-foreground" href={issuesUrl}>
            Issues
          </a>
        </nav>
      </div>
      <div className="mt-8 text-center sm:text-left">
        <p className="text-[12px]">© {new Date().getFullYear()} MamboRambo. Built for privacy and speed.</p>
      </div>
    </footer>
  )
}

import mamboramboLogo from "@/assets/mamborambo-mark.png"
import { GithubIcon } from "@/components/icons/GithubIcon"
import { githubUrl } from "@/lib/links"

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-50 w-full">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-4 sm:px-6">
        <div className="absolute inset-0 -z-10 border-b border-border/40 bg-background/80 backdrop-blur-xl" />
        
        <a href="/" className="group inline-flex items-center gap-2.5 text-base font-bold tracking-tight text-foreground">
          <img src={mamboramboLogo} alt="" className="size-8 transition-transform group-hover:scale-110" />
          <span>MamboRambo</span>
        </a>
        
        <a
          href={githubUrl}
          target="_blank"
          rel="noreferrer"
          className="inline-flex size-10 items-center justify-center rounded-2xl text-muted-foreground transition-all hover:bg-white hover:text-foreground hover:shadow-sm hover:ring-1 hover:ring-black/[0.03]"
          aria-label="MamboRambo on GitHub"
        >
          <GithubIcon className="size-5" />
        </a>
      </div>
    </header>
  )
}

import mamboramboLogo from "@/assets/mamborambo-logo.png"
import { DownloadCta } from "@/components/DownloadCta"
import { SiteFooter } from "@/components/SiteFooter"
import { SiteHeader } from "@/components/SiteHeader"
import { StudioPreview } from "@/components/StudioPreview"

export function HomePage() {
  return (
    <div className="relative flex min-h-svh flex-col overflow-hidden bg-background selection:bg-primary selection:text-primary-foreground">
      {/* Background Decoration */}
      <div className="pointer-events-none absolute top-0 left-1/2 -z-10 h-[800px] w-full -translate-x-1/2 -translate-y-1/2 opacity-25 [background:radial-gradient(circle_at_center,var(--color-primary)_0%,transparent_70%)] blur-[100px]" />
      
      <SiteHeader />

      <main className="flex-1">
        {/* Refined, Elegant Hero */}
        <section className="mx-auto flex w-full max-w-5xl flex-col items-center px-4 pt-8 pb-8 text-center sm:px-6 sm:pt-12 sm:pb-12">
          <div className="mb-4 transition-transform hover:scale-105 duration-500">
            <img src={mamboramboLogo} alt="MamboRambo" className="h-auto w-72 object-contain" />
          </div>
          
          <h1 className="max-w-4xl text-5xl font-bold tracking-tight text-balance text-foreground sm:text-6xl md:text-7xl">
            Offline text to speech <span className="text-muted-foreground/60 font-medium italic">for your desktop.</span>
          </h1>
          
          <p className="mt-5 max-w-2xl text-lg leading-relaxed text-muted-foreground sm:text-xl">
            MamboRambo runs locally, generates natural speech, and keeps your audio workflow on your machine.
          </p>

          <div className="mt-6 w-full flex justify-center">
            <DownloadCta />
          </div>
        </section>

        {/* Elegant Feature Section - Simple & airy */}
        <section className="mx-auto max-w-5xl px-4 py-16 sm:px-6">
            <div className="grid gap-6 sm:grid-cols-3">
                {[
                    { title: "Total Privacy", desc: "Your data never leaves your machine. Your audio, your rules." },
                    { title: "Languages", desc: "Generate speech in English, Chinese, Japanese, and more." },
                    { title: "Production Ready", desc: "High-fidelity models that fit into your creative workflow." }
                ].map((feat) => (
                    <div key={feat.title} className="group relative rounded-3xl bg-white/30 p-8 border border-border/40 transition-all hover:bg-white/60 hover:border-border/60 hover:-translate-y-1 duration-500">
                        <h3 className="relative font-bold text-foreground mb-3 text-[11px] uppercase tracking-[0.2em]">{feat.title}</h3>
                        <p className="relative text-muted-foreground text-[14px] leading-relaxed font-medium">{feat.desc}</p>
                    </div>
                ))}
            </div>
        </section>

        <StudioPreview />
      </main>

      <SiteFooter />
    </div>
  )
}

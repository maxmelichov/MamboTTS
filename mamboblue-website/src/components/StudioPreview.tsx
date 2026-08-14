import studioPreview from "@/assets/studio-preview.png"

export function StudioPreview() {
  return (
    <section className="mx-auto w-full max-w-7xl px-4 py-12 sm:px-6 sm:py-18">
      <div className="grid items-center gap-10 lg:grid-cols-[0.72fr_1.28fr]">
        <div className="max-w-xl">
          <p className="text-[11px] font-bold uppercase tracking-[0.24em] text-muted-foreground">
            Desktop Studio
          </p>
          <h2 className="mt-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
            Write, generate, and preview speech in one quiet workspace.
          </h2>
          <p className="mt-4 text-base leading-7 text-muted-foreground sm:text-lg">
            MamboRambo keeps the flow simple: type the line, generate locally, then
            play back or export the result without leaving the app.
          </p>
        </div>

        <div className="relative">
          <div className="absolute inset-x-8 bottom-0 -z-10 h-30 rounded-full bg-black/15 blur-3xl" />
          <img
            src={studioPreview}
            alt="MamboRambo desktop studio with text input, generate button, and audio preview"
            className="w-full rounded-[1.75rem] border border-white/70 shadow-2xl shadow-black/20 sm:rounded-[2.25rem]"
            loading="lazy"
          />
        </div>
      </div>
    </section>
  )
}

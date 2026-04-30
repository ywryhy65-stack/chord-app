export default function HomePage() {
  return (
    <main className="min-h-screen bg-background text-foreground flex items-center justify-center px-4 py-10">
      <div className="max-w-2xl rounded-3xl border border-border bg-card/80 p-10 shadow-xl shadow-black/10 backdrop-blur-md">
        <h1 className="text-4xl font-bold tracking-tight">Chord App</h1>
        <p className="mt-4 text-lg leading-8 text-muted-foreground">
          The app is now configured and ready. Start building your chords interface here.
        </p>
      </div>
    </main>
  )
}

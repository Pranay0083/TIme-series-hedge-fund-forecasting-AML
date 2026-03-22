export default function TitleSlide() {
  return (
    <div className="flex items-center justify-center h-full px-8">
      <div className="text-center max-w-4xl">
        <div className="fade-up opacity-0 stagger-1">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-primary/10 border border-primary/20 mb-8">
            <span className="w-2 h-2 rounded-full bg-accent animate-pulse" />
            <span className="text-xs font-mono text-accent tracking-wider uppercase">Advanced Machine Learning</span>
          </div>
        </div>

        <h1 className="text-6xl md:text-7xl lg:text-8xl font-bold mb-6 fade-up opacity-0 stagger-2 leading-tight">
          <span className="gradient-text">Hedge Fund</span>
          <br />
          <span className="text-text-primary">Time-Series Forecasting</span>
        </h1>

        <p className="text-lg md:text-xl text-text-secondary max-w-2xl mx-auto mb-12 fade-up opacity-0 stagger-3">
          Multi-horizon return prediction using LightGBM ensembles,
          symbolic regression feature discovery, and temporal weight decay
        </p>

        <div className="flex flex-wrap justify-center gap-4 fade-up opacity-0 stagger-4">
          {['LightGBM', 'Symbolic Regression', 'tsfresh', 'Multi-Horizon'].map((tag) => (
            <span key={tag} className="px-4 py-2 rounded-lg bg-surface-lighter border border-border text-sm text-text-secondary font-mono">
              {tag}
            </span>
          ))}
        </div>

        <div className="mt-16 fade-up opacity-0 stagger-5">
          <div className="inline-flex items-center gap-6 text-sm text-text-muted">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center">
                <svg className="w-4 h-4 text-primary-light" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
              </div>
              <span>Team TLQ</span>
            </div>
            <div className="w-px h-4 bg-border" />
            <span className="font-mono text-xs">Spring 2026</span>
            <div className="w-px h-4 bg-border" />
            <span className="font-mono text-xs">12 Slides</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function ProblemSlide() {
  return (
    <div className="flex items-center justify-center h-full px-8">
      <div className="max-w-5xl w-full">
        <div className="fade-up opacity-0 stagger-1 mb-2">
          <span className="text-xs font-mono text-accent tracking-wider uppercase">02 / Problem Statement</span>
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-4 fade-up opacity-0 stagger-1">
          <span className="gradient-text">Why This Matters</span>
        </h2>
        <p className="text-text-secondary text-lg mb-10 fade-up opacity-0 stagger-2 max-w-3xl">
          Predicting hedge fund returns is a critical challenge in quantitative finance. 
          Accurate multi-horizon forecasts enable better risk management, capital allocation, 
          and alpha generation.
        </p>

        <div className="grid md:grid-cols-3 gap-6">
          {[
            {
              icon: (
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
              ),
              title: 'Multi-Horizon Prediction',
              desc: 'Forecast returns at 1, 3, 10, and 25-day horizons simultaneously — each with distinct signal-to-noise characteristics.',
              delay: 'stagger-2',
            },
            {
              icon: (
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              ),
              title: 'Weighted RMSE Metric',
              desc: 'Competition metric emphasizes prediction accuracy where it matters most — samples carry different importance weights.',
              delay: 'stagger-3',
            },
            {
              icon: (
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 7v10c0 2 1 3 3 3h10c2 0 3-1 3-3V7c0-2-1-3-3-3H7C5 4 4 5 4 7z M9 2v3M15 2v3M4 10h16" />
                </svg>
              ),
              title: 'Temporal Ordering',
              desc: 'Financial time-series data requires careful treatment — no look-ahead bias, proper time-based validation splits with gaps.',
              delay: 'stagger-4',
            },
          ].map((card) => (
            <div key={card.title} className={`glass-card p-6 fade-up opacity-0 ${card.delay} hover:border-primary/30 transition-all duration-300`}>
              <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center text-primary-light mb-4">
                {card.icon}
              </div>
              <h3 className="text-lg font-semibold text-text-primary mb-2">{card.title}</h3>
              <p className="text-sm text-text-secondary leading-relaxed">{card.desc}</p>
            </div>
          ))}
        </div>

        <div className="mt-8 glass-card p-5 fade-up opacity-0 stagger-5">
          <div className="flex items-start gap-4">
            <div className="w-10 h-10 rounded-lg bg-warning/10 flex items-center justify-center shrink-0">
              <svg className="w-5 h-5 text-warning" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
            </div>
            <div>
              <h4 className="text-sm font-semibold text-warning mb-1">Key Challenge</h4>
              <p className="text-sm text-text-secondary">
                90+ anonymized features with varying relevance across horizons, 5 asset sub-categories,
                heavy-tailed return distributions, and non-stationary temporal dynamics.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

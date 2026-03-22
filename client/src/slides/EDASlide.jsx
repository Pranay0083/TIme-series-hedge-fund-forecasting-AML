export default function EDASlide() {
  const findings = [
    {
      title: 'Missing Data Patterns',
      detail: 'Several features have >30% missing values with non-random patterns — missingness itself is a signal. Created binary missing flags.',
      icon: '🔍',
    },
    {
      title: 'Heavy-Tailed Returns',
      detail: 'Target (y_target) exhibits heavy tails with kurtosis far exceeding Gaussian. Applied target clipping and horizon-specific normalization.',
      icon: '📊',
    },
    {
      title: 'Horizon Heterogeneity',
      detail: 'Feature importance varies dramatically across horizons (1, 3, 10, 25 days) — motivating per-horizon model training.',
      icon: '📈',
    },
    {
      title: 'Temporal Non-Stationarity',
      detail: 'Feature distributions shift over time. Recent data is more predictive than older data — motivating temporal weight decay.',
      icon: '⏰',
    },
    {
      title: 'Category Structure',
      detail: '5 sub-categories with distinct statistical profiles. Feature relevance differs by sub-category, suggesting granular models.',
      icon: '🏷️',
    },
    {
      title: 'Feature Correlations',
      detail: 'Top features (feature_al, feature_am, feature_bz, feature_s) show high mutual information with target across all horizons.',
      icon: '🔗',
    },
  ]

  return (
    <div className="flex items-center justify-center h-full px-8">
      <div className="max-w-5xl w-full">
        <div className="fade-up opacity-0 stagger-1 mb-2">
          <span className="text-xs font-mono text-accent tracking-wider uppercase">04 / Exploratory Analysis</span>
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-8 fade-up opacity-0 stagger-1">
          <span className="gradient-text">Key EDA Findings</span>
        </h2>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {findings.map((f, i) => (
            <div key={f.title}
              className={`glass-card p-5 fade-up opacity-0 stagger-${i + 2} hover:border-primary/30 transition-all duration-300 group`}>
              <div className="text-2xl mb-3">{f.icon}</div>
              <h3 className="text-base font-semibold text-text-primary mb-2 group-hover:text-primary-light transition-colors">
                {f.title}
              </h3>
              <p className="text-sm text-text-secondary leading-relaxed">{f.detail}</p>
            </div>
          ))}
        </div>

        <div className="mt-6 glass-card p-4 fade-up opacity-0 stagger-6">
          <div className="flex items-center gap-3 text-sm">
            <div className="w-8 h-8 rounded-lg bg-success/10 flex items-center justify-center">
              <svg className="w-4 h-4 text-success" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <span className="text-text-secondary">
              EDA performed across <span className="text-text-primary font-semibold">6 notebooks</span> including tsfresh automated feature extraction and filtering
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

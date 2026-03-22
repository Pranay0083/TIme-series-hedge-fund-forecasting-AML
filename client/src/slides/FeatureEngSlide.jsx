export default function FeatureEngSlide() {
  const featureGroups = [
    {
      name: 'Target Encoding',
      count: 2,
      desc: 'sub_category, sub_code mean-encoded against y_target (train-only to avoid leakage)',
      color: 'border-l-primary',
    },
    {
      name: 'Lag Features',
      count: 18,
      desc: 'Shifts of 1, 3, 10 steps for top 6 features grouped by (code, sub_code, sub_category, horizon)',
      color: 'border-l-accent',
    },
    {
      name: 'Rolling Statistics',
      count: 24,
      desc: 'Rolling mean and std with windows 5, 10 for top features — captures local trends and volatility',
      color: 'border-l-success',
    },
    {
      name: 'EWM Features',
      count: 6,
      desc: 'Exponential weighted mean (span=5) — recent values weighted exponentially more',
      color: 'border-l-warning',
    },
    {
      name: 'Cross-Sectional Ranks',
      count: 6,
      desc: 'Percentile rank within each time step — captures relative position across assets',
      color: 'border-l-danger',
    },
    {
      name: 'Interaction Features',
      count: 7,
      desc: 'Differences, ratios, and products of top correlated features (al-am, cg-by, temporal cycle)',
      color: 'border-l-primary-light',
    },
  ]

  return (
    <div className="flex items-center justify-center h-full px-8">
      <div className="max-w-5xl w-full">
        <div className="fade-up opacity-0 stagger-1 mb-2">
          <span className="text-xs font-mono text-accent tracking-wider uppercase">06 / Feature Engineering</span>
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-2 fade-up opacity-0 stagger-1">
          <span className="gradient-text">Feature Engineering</span>
        </h2>
        <p className="text-text-secondary mb-8 fade-up opacity-0 stagger-2">
          Systematic construction of temporal, cross-sectional, and interaction features
        </p>

        <div className="grid md:grid-cols-2 gap-4">
          {featureGroups.map((g, i) => (
            <div key={g.name}
              className={`glass-card p-4 border-l-2 ${g.color} fade-up opacity-0 stagger-${i + 2} hover:bg-surface-lighter/30 transition-all`}>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-semibold text-text-primary">{g.name}</h3>
                <span className="font-mono text-xs px-2 py-0.5 rounded-full bg-surface-lighter text-accent">{g.count} feats</span>
              </div>
              <p className="text-xs text-text-secondary leading-relaxed">{g.desc}</p>
            </div>
          ))}
        </div>

        <div className="mt-6 glass-card p-5 fade-up opacity-0 stagger-6">
          <h3 className="text-sm font-semibold text-text-primary mb-3">Top 6 Base Features (Highest IC)</h3>
          <div className="flex flex-wrap gap-2">
            {['feature_al', 'feature_am', 'feature_cg', 'feature_by', 'feature_s', 'feature_bz'].map((f) => (
              <code key={f} className="text-xs font-mono px-3 py-1.5 rounded-lg bg-primary/10 text-primary-light border border-primary/20">
                {f}
              </code>
            ))}
          </div>
          <p className="text-xs text-text-muted mt-3">
            These features form the basis for lag, rolling, EWM, rank, and interaction feature derivation
          </p>
        </div>
      </div>
    </div>
  )
}

export default function PreprocessSlide() {
  const steps = [
    {
      step: '01',
      title: 'Feature Dropping',
      desc: 'Remove low-variance & 100% missing features (e.g. feature_b to feature_g).',
      reason: 'These features act as pure noise; keeping them inflates tree split search times without providing any signal.',
      color: 'text-danger',
      bg: 'bg-danger/10',
    },
    {
      step: '02',
      title: 'Missing Flags',
      desc: 'Create binary indicators (_is_missing) for high-missing features.',
      reason: 'In quantitative finance, the absence of data is often systematic (e.g., untraded assets), which provides predictive signal.',
      color: 'text-warning',
      bg: 'bg-warning/10',
    },
    {
      step: '03',
      title: 'Hierarchical Imputation',
      desc: 'Group-level median fill (code × sub_category) → global median fallback.',
      reason: 'Assets within the same group share financial properties. Group medians prevent structural leakage across distinct market regimes.',
      color: 'text-primary-light',
      bg: 'bg-primary/10',
    },
    {
      step: '04',
      title: 'Categorical Encoding',
      desc: 'Label encoding for categorical IDs, with explicit unknown handling (-1).',
      reason: 'Permits tree models to isolate specific non-continuous attributes cleanly without expanding into massive one-hot sparse matrices.',
      color: 'text-accent',
      bg: 'bg-accent/10',
    },
    {
      step: '05',
      title: 'Target Processing',
      desc: 'Clip extreme outliers & horizon-specific variance normalization.',
      reason: 'Prevents extreme tail events (kurtosis > 200) from dominating MSE loss, ensuring stable tree-leaf weight optimization.',
      color: 'text-success',
      bg: 'bg-success/10',
    },
    {
      step: '06',
      title: 'Weight Filtering',
      desc: 'Drop samples where row weight == 0.',
      reason: 'Zero-weight samples do not contribute to the final scoring metric. Removing them saves compute and prevents zero-contribution bias.',
      color: 'text-primary-light',
      bg: 'bg-primary/10',
    },
  ]

  return (
    <div className="flex items-center justify-center h-full px-8 overflow-y-auto">
      <div className="max-w-6xl w-full py-8">
        <div className="fade-up opacity-0 stagger-1 mb-2">
          <span className="text-xs font-mono text-accent tracking-wider uppercase">05 / Preprocessing</span>
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-8 fade-up opacity-0 stagger-1">
          <span className="gradient-text">Preprocessing Steps & Rationale</span>
        </h2>

        <div className="grid md:grid-cols-2 gap-6">
          {steps.map((s, i) => (
            <div key={s.step} className={`fade-up opacity-0 stagger-${i + 1} glass-card p-5 hover:border-${s.color.split('-')[1]}/50 transition-all flex flex-col`}>
              <div className="flex items-center gap-3 mb-3">
                <div className={`shrink-0 w-10 h-10 rounded-xl ${s.bg} flex items-center justify-center font-mono text-sm font-bold ${s.color}`}>
                  {s.step}
                </div>
                <h3 className="text-lg font-semibold text-text-primary">{s.title}</h3>
              </div>
              <p className="text-sm text-text-secondary mb-3 pb-3 border-b border-border/30 flex-grow">
                {s.desc}
              </p>
              <div>
                <span className={`text-xs font-bold ${s.color} uppercase tracking-wider block mb-1`}>Why?</span>
                <p className="text-xs text-text-muted leading-relaxed">{s.reason}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

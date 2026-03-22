export default function FeatureEng1() {
  const findings = [
    {
      title: 'Target Embargo (Gap Size = 5)',
      detail: 'Overlapping horizon targets (up to 25 days) natively create misleading mechanical autocorrelations.',
      icon: '🛡️',
      what: 'Interjected a structural embargo block (gap_size=5) directly between training and validation data spans.',
      why: 'Guarantees the model cannot peek into future temporal data. Ensures overlapping forecast horizons do not artificially inflate validation scores.'
    },
    {
      title: 'Expanding Window Validation',
      detail: 'Standard K-fold random splits cause retroactive data leakage. Expanding window historical slices significantly outperform rolling variants.',
      icon: '📆',
      what: 'Implemented GroupTimeSeriesSplit grouping directly by the ts_index with an expanding window setup.',
      why: 'A rigorous backtesting framework is the only way to trust feature engineering metrics in a non-stationary time-series environment.'
    },
    {
      title: 'Pre-TSFresh Exclusion Gates',
      detail: 'The raw dataset contains pure noise, monotonic counters, and excessively sparse columns which break automated extractors.',
      icon: '🚧',
      what: 'Executed 10 rigorous exclusion rule checks prior to extraction (variance thresholds, lag-correlation checks, density ratios).',
      why: 'Eliminates structural artifacts before generating heavy dimensionality. Feeds only pure, dynamically fluctuating signals into the tsfresh engine to save compute.'
    }
  ]

  return (
    <div className="flex items-center justify-center h-full px-8 pb-10">
      <div className="max-w-6xl w-full">
        <div className="fade-up opacity-0 stagger-1 mb-2">
          <span className="text-xs font-mono text-accent tracking-wider uppercase">05.1 / Feature Engineering I</span>
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-6 fade-up opacity-0 stagger-1">
          <span className="gradient-text">Validation Framing & Signal Filtering</span>
        </h2>

        <div className="grid md:grid-cols-3 gap-6 overflow-y-auto max-h-[60vh] p-2 hide-scrollbar">
          {findings.map((f, i) => (
            <div key={f.title} className={`glass-card p-5 fade-up opacity-0 stagger-${i + 2} hover:border-primary/50 transition-all duration-300 group flex flex-col`}>
              <div className="flex items-center gap-3 mb-3">
                <div className="text-3xl bg-secondary/20 p-2 rounded-lg">{f.icon}</div>
                <h3 className="text-lg font-bold text-text-primary group-hover:text-primary-light transition-colors">{f.title}</h3>
              </div>
              <p className="text-sm border-l-2 border-accent/50 pl-3 mb-4 text-text-secondary leading-relaxed flex-grow">{f.detail}</p>
              
              <div className="space-y-3 mt-auto pt-4 border-t border-white/5">
                <div>
                  <span className="text-xs font-bold text-accent uppercase tracking-wider block mb-1">What & How?</span>
                  <p className="text-xs text-text-secondary">{f.what}</p>
                </div>
                <div>
                  <span className="text-xs font-bold text-success uppercase tracking-wider block mb-1">Why do we care?</span>
                  <p className="text-xs text-text-secondary">{f.why}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
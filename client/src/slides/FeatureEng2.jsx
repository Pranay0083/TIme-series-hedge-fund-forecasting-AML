export default function FeatureEng2() {
  const findings = [
    {
      title: 'Automated Extraction (TSFresh)',
      detail: 'Automated extractions produce massive feature dimensionality, but most are statistically overlapping.',
      icon: '⚙️',
      what: 'Utilized tsfresh to generate large-scale time-series feature statistics on the pre-filtered columns.',
      why: 'Captures complex non-linear temporal dynamics (e.g. wavelet transforms, entropy, Fourier coefficients) that manual engineering would miss.'
    },
    {
      title: 'Sequential 3-Gate Pruning',
      detail: 'Drastically reduced dimensionality by pruning features that fail strict statistical relevance checks.',
      icon: '✂️',
      what: 'Applied 1. Hypothesis Testing (p-value < 0.05), 2. Redundancy Control (|corr| > 0.9), and 3. Temporal Fold Continuity tests.',
      why: 'Countered the curse of dimensionality. Locked the final predictive set to only represent statistics proving fundamentally relevant and immune to isolated regimes.'
    },
    {
      title: 'Data Preprocessing pipeline',
      detail: '48 features suffered from missing values (up to 12.5%).',
      icon: '🛠️',
      what: 'Implemented group-aware median imputation (by code × sub_category) combined with explicit _is_missing boolean flags.',
      why: 'Mitigates missing data biases accurately across entities without generating test leakage, allowing tree models to leverage missingness as a signal.'
    }
  ]

  return (
    <div className="flex items-center justify-center h-full px-8 pb-10">
      <div className="max-w-6xl w-full">
        <div className="fade-up opacity-0 stagger-1 mb-2">
          <span className="text-xs font-mono text-accent tracking-wider uppercase">05.2 / Feature Engineering II</span>
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-6 fade-up opacity-0 stagger-1">
          <span className="gradient-text">Generation, Pruning & Preprocessing</span>
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
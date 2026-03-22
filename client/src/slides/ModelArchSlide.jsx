export default function ModelArchSlide() {
  const config = [
    { param: 'objective', value: 'regression' },
    { param: 'learning_rate', value: '0.02' },
    { param: 'n_estimators', value: '6,000' },
    { param: 'num_leaves', value: '96' },
    { param: 'min_child_samples', value: '150' },
    { param: 'feature_fraction', value: '0.7' },
    { param: 'bagging_fraction', value: '0.8' },
    { param: 'lambda_l1', value: '0.2' },
    { param: 'lambda_l2', value: '15.0' },
    { param: 'early_stopping', value: '200 rounds' },
  ]

  return (
    <div className="flex items-center justify-center h-full px-8">
      <div className="max-w-5xl w-full">
        <div className="fade-up opacity-0 stagger-1 mb-2">
          <span className="text-xs font-mono text-accent tracking-wider uppercase">08 / Model Architecture</span>
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-8 fade-up opacity-0 stagger-1">
          <span className="gradient-text">LightGBM Ensemble</span>
        </h2>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Architecture diagram */}
          <div className="fade-up opacity-0 stagger-2">
            <div className="glass-card p-6">
              <h3 className="text-sm font-semibold text-text-primary mb-4 flex items-center gap-2">
                <svg className="w-4 h-4 text-primary-light" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
                Training Architecture
              </h3>

              <div className="space-y-3">
                {/* Data split */}
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-8 rounded-lg bg-primary/20 flex items-center justify-center text-xs font-mono text-primary-light">
                    Train (ts_index ≤ 3500)
                  </div>
                  <div className="w-px h-6 bg-border" />
                  <div className="w-32 h-8 rounded-lg bg-accent/20 flex items-center justify-center text-xs font-mono text-accent">
                    Val (&gt; 3500)
                  </div>
                </div>

                <div className="text-center text-text-muted text-xs">↓ per-horizon split</div>

                {/* Per horizon models */}
                <div className="grid grid-cols-4 gap-2">
                  {[1, 3, 10, 25].map((h) => (
                    <div key={h} className="p-2 rounded-lg bg-surface-lighter border border-border text-center">
                      <div className="text-[10px] text-text-muted">Horizon</div>
                      <div className="text-sm font-bold font-mono text-primary-light">{h}</div>
                    </div>
                  ))}
                </div>

                <div className="text-center text-text-muted text-xs">↓ 5-seed ensemble</div>

                {/* Seeds */}
                <div className="flex justify-center gap-2">
                  {[42, 2024, 7, 11, 999].map((s) => (
                    <div key={s} className="px-2 py-1 rounded bg-success/10 text-[10px] font-mono text-success">
                      seed={s}
                    </div>
                  ))}
                </div>

                <div className="text-center text-text-muted text-xs">↓ average predictions</div>

                <div className="p-3 rounded-lg bg-gradient-to-r from-primary/20 to-accent/20 border border-primary/20 text-center">
                  <div className="text-xs font-semibold text-text-primary">Final Ensemble Prediction</div>
                </div>
              </div>
            </div>
          </div>

          {/* Config table */}
          <div className="fade-up opacity-0 stagger-3">
            <div className="glass-card p-1">
              <div className="px-4 py-3 border-b border-border">
                <h3 className="text-sm font-semibold text-text-primary flex items-center gap-2">
                  <svg className="w-4 h-4 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  LightGBM Hyperparameters
                </h3>
              </div>
              <div className="divide-y divide-border/50">
                {config.map((c) => (
                  <div key={c.param} className="px-4 py-2.5 flex items-center justify-between hover:bg-surface-lighter/50 transition-colors">
                    <code className="text-xs font-mono text-text-secondary">{c.param}</code>
                    <span className="text-sm font-mono text-accent font-semibold">{c.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

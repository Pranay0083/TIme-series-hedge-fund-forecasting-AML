import React from 'react'

export default function AdvancedSlide() {
  return (
    <div className="flex items-center justify-center h-full px-8">
      <div className="max-w-5xl w-full">
        <div className="fade-up opacity-0 stagger-1 mb-2">
          <span className="text-xs font-mono text-accent tracking-wider uppercase">09 / Advanced Techniques</span>
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-8 fade-up opacity-0 stagger-1">
          <span className="gradient-text">Advanced Strategies</span>
        </h2>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Weight Decay */}
          <div className="glass-card p-6 fade-up opacity-0 stagger-2 hover:border-primary/30 transition-all">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
                <svg className="w-5 h-5 text-primary-light" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-text-primary">Temporal Weight Decay</h3>
            </div>
            <p className="text-sm text-text-secondary mb-4">
              Recent data is more predictive. Training sample weights are multiplied by exponential decay factors:
            </p>
            <div className="bg-surface rounded-lg p-3 font-mono text-xs mb-4">
              <div className="text-text-secondary">decay = <span className="text-primary-light">exp</span>(-λ × (max_ts - ts_i))</div>
              <div className="text-text-muted mt-1">where λ = <span className="text-accent">ln(2) / half_life</span></div>
            </div>
            <div className="grid grid-cols-4 gap-2">
              {[
                { h: 1, hl: 500 },
                { h: 3, hl: 700 },
                { h: 10, hl: 1000 },
                { h: 25, hl: 1500 },
              ].map((d) => (
                <div key={d.h} className="text-center p-2 rounded-lg bg-surface-lighter">
                  <div className="text-[10px] text-text-muted">H={d.h}</div>
                  <div className="text-sm font-mono font-bold text-accent">{d.hl}</div>
                  <div className="text-[10px] text-text-muted">half-life</div>
                </div>
              ))}
            </div>
          </div>

          {/* Granular Models */}
          <div className="glass-card p-6 fade-up opacity-0 stagger-3 hover:border-primary/30 transition-all">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-accent/10 flex items-center justify-center">
                <svg className="w-5 h-5 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-text-primary">Granular Models</h3>
            </div>
            <p className="text-sm text-text-secondary mb-4">
              Beyond per-horizon training, we explored dedicated models for every
              (sub_category × horizon) pair:
            </p>
            <div className="bg-surface rounded-lg p-3 mb-4">
              <div className="grid grid-cols-5 gap-1 text-center">
                <div className="text-[10px] text-text-muted" />
                {[1, 3, 10, 25].map((h) => (
                  <div key={h} className="text-[10px] font-mono text-accent">H={h}</div>
                ))}
                {['SC₁', 'SC₂', 'SC₃', 'SC₄', 'SC₅'].map((sc) => (
                  <React.Fragment key={sc}>
                    <div className="text-[10px] text-text-muted text-right pr-2">{sc}</div>
                    {[1, 3, 10, 25].map((h) => (
                      <div key={`${sc}-${h}`} className="w-6 h-6 rounded bg-primary/20 border border-primary/10 mx-auto flex items-center justify-center">
                        <span className="text-[8px] text-primary-light">M</span>
                      </div>
                    ))}
                  </React.Fragment>
                ))}
              </div>
            </div>
            <div className="text-center">
              <span className="text-sm font-mono text-text-primary">5 × 4 = </span>
              <span className="text-lg font-bold font-mono gradient-text">20 independent models</span>
            </div>
          </div>

          {/* tsfresh */}
          <div className="glass-card p-6 fade-up opacity-0 stagger-4 hover:border-primary/30 transition-all">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-success/10 flex items-center justify-center">
                <svg className="w-5 h-5 text-success" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-text-primary">tsfresh Feature Extraction</h3>
            </div>
            <p className="text-sm text-text-secondary mb-3">
              Automated extraction of hundreds of time-series features (entropy, autocorrelation, 
              quantiles, etc.) followed by statistical significance filtering.
            </p>
            <div className="flex gap-2">
              <span className="text-xs px-2 py-1 rounded bg-success/10 text-success font-mono">extraction</span>
              <span className="text-xs px-2 py-1 rounded bg-success/10 text-success font-mono">filtering</span>
              <span className="text-xs px-2 py-1 rounded bg-success/10 text-success font-mono">selection</span>
            </div>
          </div>

          {/* Time-Series CV */}
          <div className="glass-card p-6 fade-up opacity-0 stagger-5 hover:border-primary/30 transition-all">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-warning/10 flex items-center justify-center">
                <svg className="w-5 h-5 text-warning" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-text-primary">Time-Series Cross-Validation</h3>
            </div>
            <p className="text-sm text-text-secondary mb-3">
              GroupTimeSeriesSplit with gap to prevent leakage. Expanding window,
              5-fold, with Spearman rank correlation and weighted RMSE per fold.
            </p>
            <div className="space-y-1">
              {[1, 2, 3, 4, 5].map((fold) => (
                <div key={fold} className="flex items-center gap-1">
                  <div style={{ width: `${40 + fold * 8}%` }} className="h-4 rounded bg-primary/30 flex items-center justify-end pr-2">
                    <span className="text-[8px] font-mono text-text-muted">train</span>
                  </div>
                  <div className="w-2 h-4 bg-surface-lighter rounded" />
                  <div className="h-4 rounded bg-accent/30 flex-1 flex items-center justify-center">
                    <span className="text-[8px] font-mono text-text-muted">val</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

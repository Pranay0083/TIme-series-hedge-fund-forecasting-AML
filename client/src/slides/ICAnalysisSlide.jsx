export default function ICAnalysisSlide() {
  const groups = [
    { name: 'raw_signals', h1: 0.043, h3: 0.038, h10: 0.032, h25: 0.028 },
    { name: 'discovered_gp_psr', h1: 0.041, h3: 0.039, h10: 0.035, h25: 0.031 },
    { name: 'lag_features', h1: 0.035, h3: 0.033, h10: 0.030, h25: 0.026 },
    { name: 'rolling_features', h1: 0.032, h3: 0.030, h10: 0.028, h25: 0.025 },
    { name: 'rank_features', h1: 0.030, h3: 0.028, h10: 0.025, h25: 0.022 },
    { name: 'category_enc', h1: 0.022, h3: 0.020, h10: 0.018, h25: 0.016 },
    { name: 'volatility_stats', h1: 0.028, h3: 0.026, h10: 0.024, h25: 0.021 },
    { name: 'polynomial', h1: 0.025, h3: 0.024, h10: 0.022, h25: 0.020 },
  ]

  const getHeatColor = (val) => {
    if (val >= 0.04) return 'bg-primary/60 text-white'
    if (val >= 0.035) return 'bg-primary/40 text-white'
    if (val >= 0.03) return 'bg-primary/25 text-text-primary'
    if (val >= 0.025) return 'bg-primary/15 text-text-secondary'
    return 'bg-primary/8 text-text-muted'
  }

  return (
    <div className="flex items-center justify-center h-full px-8">
      <div className="max-w-5xl w-full">
        <div className="fade-up opacity-0 stagger-1 mb-2">
          <span className="text-xs font-mono text-accent tracking-wider uppercase">10 / IC Analysis</span>
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-2 fade-up opacity-0 stagger-1">
          <span className="gradient-text">Information Coefficient</span>
        </h2>
        <p className="text-text-secondary mb-8 fade-up opacity-0 stagger-2">
          IC/Rank-IC analysis measures the predictive power of each feature group per horizon
        </p>

        {/* IC Heatmap */}
        <div className="glass-card p-6 fade-up opacity-0 stagger-3">
          <h3 className="text-sm font-semibold text-text-primary mb-4">Mean |IC| by Feature Group × Horizon</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr>
                  <th className="text-left py-2 px-3 text-text-muted font-mono text-xs">Feature Group</th>
                  {[1, 3, 10, 25].map((h) => (
                    <th key={h} className="text-center py-2 px-3 text-accent font-mono text-xs">H={h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30">
                {groups.map((g) => (
                  <tr key={g.name} className="hover:bg-surface-lighter/30 transition-colors">
                    <td className="py-2 px-3 font-mono text-xs text-text-secondary">{g.name}</td>
                    {[g.h1, g.h3, g.h10, g.h25].map((v, i) => (
                      <td key={i} className="text-center py-2 px-3">
                        <span className={`inline-block px-2 py-1 rounded text-xs font-mono ${getHeatColor(v)}`}>
                          {v.toFixed(3)}
                        </span>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-4 mt-5 fade-up opacity-0 stagger-4">
          <div className="glass-card p-4">
            <h4 className="text-xs text-text-muted uppercase tracking-wider mb-2">Key Insight 1</h4>
            <p className="text-sm text-text-secondary">
              <span className="text-text-primary font-semibold">IC decays with horizon</span> — shorter-term predictions are more predictable
            </p>
          </div>
          <div className="glass-card p-4">
            <h4 className="text-xs text-text-muted uppercase tracking-wider mb-2">Key Insight 2</h4>
            <p className="text-sm text-text-secondary">
              <span className="text-text-primary font-semibold">Discovered features match raw signals</span> — symbolic regression captures similar predictive power with fewer features
            </p>
          </div>
          <div className="glass-card p-4">
            <h4 className="text-xs text-text-muted uppercase tracking-wider mb-2">Key Insight 3</h4>
            <p className="text-sm text-text-secondary">
              <span className="text-text-primary font-semibold">Weighted IC</span> with temporal decay improves signal identification for model training
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

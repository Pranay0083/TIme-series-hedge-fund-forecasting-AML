export default function SymbolicRegressionSlide() {
  const discoveries = [
    { horizon: 'Combined', tool: 'GPlearn', formula: 'feature_l × feature_ca × feature_t', label: 'gp_l_ca_t' },
    { horizon: 'Combined', tool: 'PySR', formula: 'feature_bz × feature_t²', label: 'psr_bz_t2' },
    { horizon: 'H=1', tool: 'GPlearn', formula: '(feature_cd / feature_c) × feature_q', label: 'h1_cd_div_c_q' },
    { horizon: 'H=1', tool: 'PySR', formula: '-7.47 / (2.74 - feature_aw)', label: 'h1_aw_inv' },
    { horizon: 'H=3', tool: 'GPlearn', formula: '3 × feature_bz', label: 'h3_bz_triple' },
    { horizon: 'H=3', tool: 'PySR', formula: 'feature_bz × feature_f interaction', label: 'h3_bz_f_ratio' },
    { horizon: 'H=10', tool: 'GPlearn', formula: '(feature_ah + feature_w × feature_bv) × feature_bz', label: 'h10_complex' },
    { horizon: 'H=10', tool: 'PySR', formula: 'feature_z × feature_bz²', label: 'h10_z_bz2' },
    { horizon: 'H=25', tool: 'GPlearn', formula: 'feature_b × feature_bz', label: 'h25_b_bz' },
    { horizon: 'H=25', tool: 'PySR', formula: 'feature_s × (feature_bz + 1.48)', label: 'h25_s_bz_shift' },
  ]

  return (
    <div className="flex items-center justify-center h-full px-8">
      <div className="max-w-5xl w-full">
        <div className="fade-up opacity-0 stagger-1 mb-2">
          <span className="text-xs font-mono text-accent tracking-wider uppercase">07 / Symbolic Regression</span>
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-2 fade-up opacity-0 stagger-1">
          <span className="gradient-text">Discovered Features</span>
        </h2>
        <p className="text-text-secondary mb-6 fade-up opacity-0 stagger-2 max-w-3xl">
          GPlearn and PySR automatically discovered 50+ interaction features through symbolic regression — 
          mathematical expressions that capture non-linear relationships invisible to standard feature engineering.
        </p>

        <div className="grid md:grid-cols-2 gap-3 fade-up opacity-0 stagger-3">
          {discoveries.map((d, i) => (
            <div key={d.label} className="glass-card px-4 py-3 flex items-start gap-3 hover:border-primary/20 transition-all">
              <div className="shrink-0 flex flex-col items-center gap-1">
                <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                  d.tool === 'GPlearn' ? 'bg-success/10 text-success' : 'bg-primary/10 text-primary-light'
                }`}>{d.tool}</span>
                <span className="text-[10px] font-mono text-text-muted">{d.horizon}</span>
              </div>
              <div className="flex-1 min-w-0">
                <code className="text-xs font-mono text-accent block mb-0.5">{d.label}</code>
                <span className="text-xs text-text-secondary">{d.formula}</span>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-5 grid grid-cols-3 gap-4 fade-up opacity-0 stagger-4">
          <div className="glass-card p-4 text-center">
            <div className="text-2xl font-bold font-mono gradient-text mb-1">50+</div>
            <div className="text-xs text-text-muted">Discovered Features</div>
          </div>
          <div className="glass-card p-4 text-center">
            <div className="text-2xl font-bold font-mono gradient-text mb-1">2</div>
            <div className="text-xs text-text-muted">Discovery Tools (GPlearn, PySR)</div>
          </div>
          <div className="glass-card p-4 text-center">
            <div className="text-2xl font-bold font-mono gradient-text mb-1">L1</div>
            <div className="text-xs text-text-muted">Regularization for Selection</div>
          </div>
        </div>
      </div>
    </div>
  )
}

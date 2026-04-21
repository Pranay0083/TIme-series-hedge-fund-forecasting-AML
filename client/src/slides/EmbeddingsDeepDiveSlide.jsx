export default function EmbeddingsDeepDiveSlide() {
  const categoricals = [
    { name: 'code', desc: 'Asset identifier', card: 'High (~100s)', why: 'Each asset may have unique return characteristics; embedding captures latent asset style (momentum, mean-reversion, etc.)' },
    { name: 'sub_code', desc: 'Asset sub-grouping', card: 'Medium (~50)', why: 'Assets within same sub_code share sector/strategy — embedding captures intra-group similarity' },
    { name: 'sub_category', desc: 'Asset category', card: 'Low (5)', why: 'Broad strategy class (equity, credit, etc.) — embedding learns category-level risk premia patterns' },
    { name: 'feature_a', desc: 'Encoded categorical', card: 'Low', why: 'Discrete coded feature — embedding captures ordinal or non-linear relationships trees might miss' },
    { name: 'horizon', desc: 'Forecast horizon', card: '4 (1,3,10,25)', why: 'Different horizons have different signal dynamics — embedding lets the model condition predictions on horizon-specific patterns' },
    { name: 'feature_ch', desc: 'Integer categorical', card: 'Varies', why: 'Treated as categorical (not ordinal) — embedding prevents imposing false linear ordering on non-ordinal values' },
  ]

  return (
    <div className="flex items-center justify-center h-full px-8">
      <div className="max-w-5xl w-full">
        <div className="fade-up opacity-0 stagger-1 mb-2">
          <span className="text-xs font-mono text-accent tracking-wider uppercase">Phase 2 / Entity Embeddings</span>
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-2 fade-up opacity-0 stagger-1">
          <span className="gradient-text">Entity Embeddings</span>
        </h2>
        <p className="text-text-secondary mb-6 fade-up opacity-0 stagger-2 text-sm max-w-3xl">
          Inspired by <span className="text-text-primary font-semibold">Guo & Berkhahn (2016)</span> — "Entity Embeddings of Categorical Variables" — 
          each categorical feature is mapped to a dense vector through a learned lookup table, exactly like word embeddings in NLP.
        </p>

        {/* Categorical embeddings table */}
        <div className="glass-card p-4 fade-up opacity-0 stagger-3 mb-5">
          <h3 className="text-sm font-semibold text-text-primary mb-3">Categorical Features → Embeddings</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-2 px-3 text-text-muted text-xs">Feature</th>
                  <th className="text-left py-2 px-3 text-text-muted text-xs">Description</th>
                  <th className="text-center py-2 px-3 text-accent font-mono text-xs">Cardinality</th>
                  <th className="text-left py-2 px-3 text-text-muted text-xs">Why Embed?</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/30">
                {categoricals.map((c) => (
                  <tr key={c.name} className="hover:bg-surface-lighter/30 transition-colors">
                    <td className="py-2 px-3 font-mono text-xs text-primary-light">{c.name}</td>
                    <td className="py-2 px-3 text-xs text-text-secondary">{c.desc}</td>
                    <td className="py-2 px-3 text-xs text-accent font-mono text-center">{c.card}</td>
                    <td className="py-2 px-3 text-[10px] text-text-muted leading-relaxed max-w-xs">{c.why}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-4 fade-up opacity-0 stagger-4">
          <div className="glass-card p-4">
            <h4 className="text-xs text-text-muted uppercase tracking-wider mb-2">Why Not One-Hot?</h4>
            <p className="text-xs text-text-secondary">
              <span className="text-text-primary font-semibold">One-hot creates 500+ sparse dimensions</span> — wastes memory, 
              breaks gradient flow, and treats all categories as equidistant. Embeddings learn that 
              "momentum equity" is closer to "trend-follow equity" than "credit arbitrage."
            </p>
          </div>
          <div className="glass-card p-4">
            <h4 className="text-xs text-text-muted uppercase tracking-wider mb-2">Why min(50, card//2+1)?</h4>
            <p className="text-xs text-text-secondary">
              <span className="text-text-primary font-semibold">Rule-of-thumb from Google research</span> — embedding dim should scale with 
              log(cardinality). Cap at 50 prevents overfitting on rare categories. Floor division proportionally 
              allocates capacity to information density.
            </p>
          </div>
          <div className="glass-card p-4">
            <h4 className="text-xs text-text-muted uppercase tracking-wider mb-2">Why Normal(0, 0.01) Init?</h4>
            <p className="text-xs text-text-secondary">
              <span className="text-text-primary font-semibold">Small-magnitude initialization</span> prevents any single category from dominating 
              early training. Starting near zero means embeddings must be "earned" by gradient signal — 
              categories with no predictive value stay near zero naturally.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

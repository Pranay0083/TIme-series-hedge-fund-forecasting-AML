export default function ConclusionSlide() {
  return (
    <div className="flex items-center justify-center h-full px-8">
      <div className="max-w-5xl w-full">
        <div className="fade-up opacity-0 stagger-1 mb-2">
          <span className="text-xs font-mono text-accent tracking-wider uppercase">12 / Conclusion</span>
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-8 fade-up opacity-0 stagger-1">
          <span className="gradient-text">Summary & Future Work</span>
        </h2>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Key contributions */}
          <div className="space-y-4 fade-up opacity-0 stagger-2">
            <h3 className="text-sm font-semibold text-text-primary uppercase tracking-wider flex items-center gap-2">
              <div className="w-6 h-0.5 bg-primary" />
              Key Contributions
            </h3>
            {[
              'Multi-pipeline approach: progressive refinement from baseline to advanced models',
              'Symbolic regression feature discovery with GPlearn and PySR — 50+ novel interaction features',
              'Temporal weight decay with per-horizon half-lives capturing market regime shifts',
              'Comprehensive IC/Rank-IC analysis quantifying predictive power per feature group per horizon',
              '20-model granular architecture (sub_category × horizon) for maximum specialization',
            ].map((item, i) => (
              <div key={i} className="flex items-start gap-3">
                <div className="w-5 h-5 rounded-full bg-success/10 flex items-center justify-center shrink-0 mt-0.5">
                  <svg className="w-3 h-3 text-success" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <p className="text-sm text-text-secondary leading-relaxed">{item}</p>
              </div>
            ))}
          </div>

          {/* Future work */}
          <div className="space-y-4 fade-up opacity-0 stagger-3">
            <h3 className="text-sm font-semibold text-text-primary uppercase tracking-wider flex items-center gap-2">
              <div className="w-6 h-0.5 bg-accent" />
              Future Directions
            </h3>
            {[
              {
                title: 'Information Geometry & Manifold Learning',
                desc: 'Map financial structures onto non-euclidean manifolds for robust pattern extraction in non-stationary regimes.',
              },
              {
                title: 'Target De-noising',
                desc: 'Clean robust signal streams natively by mitigating extreme fat-tail kurtosis iteratively.',
              },
              {
                title: 'Denoising Autoencoders (DAEs)',
                desc: 'Replace current hard Information Coefficient (IC) gates with automated representation learning to distill core dimensional features.',
              },
              {
                title: 'Tabular Transformers & MLP Embeddings ✅',
                desc: 'Implemented in Phase 2 — MLP with entity embeddings and FT-Transformer to capture non-linear cross-feature relationships via learned representations.',
              },
            ].map((item) => (
              <div key={item.title} className="glass-card p-4 hover:border-accent/30 transition-all">
                <h4 className="text-sm font-semibold text-text-primary mb-1">{item.title}</h4>
                <p className="text-xs text-text-secondary">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Tech stack */}
        <div className="mt-8 glass-card p-5 fade-up opacity-0 stagger-4">
          <h3 className="text-xs text-text-muted uppercase tracking-wider mb-3">Technology Stack</h3>
          <div className="flex flex-wrap gap-2">
            {['Python', 'LightGBM', 'XGBoost', 'Pandas', 'Polars', 'scikit-learn', 'GPlearn', 'PySR', 'tsfresh', 'Optuna', 'SciPy', 'NumPy'].map((tech) => (
              <span key={tech} className="px-3 py-1.5 rounded-lg bg-surface-lighter border border-border text-xs font-mono text-text-secondary">
                {tech}
              </span>
            ))}
          </div>
        </div>

        {/* Thank you */}
        <div className="mt-6 text-center fade-up opacity-0 stagger-5">
          <p className="text-lg text-text-muted">
            Thank you — <span className="gradient-text font-semibold">Team Bias and Variance</span>
          </p>
        </div>
      </div>
    </div>
  )
}

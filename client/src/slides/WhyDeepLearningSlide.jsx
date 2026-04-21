export default function WhyDeepLearningSlide() {
  return (
    <div className="flex items-center justify-center h-full px-8">
      <div className="max-w-5xl w-full">
        <div className="fade-up opacity-0 stagger-1 mb-2">
          <span className="text-xs font-mono text-accent tracking-wider uppercase">Phase 2 / Why Deep Learning?</span>
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-2 fade-up opacity-0 stagger-1">
          <span className="gradient-text">Why Move Beyond Trees?</span>
        </h2>
        <p className="text-text-secondary mb-8 fade-up opacity-0 stagger-2">
          LightGBM established a strong baseline — but it has fundamental limitations for this problem
        </p>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Limitations of LightGBM */}
          <div className="space-y-3 fade-up opacity-0 stagger-2">
            <h3 className="text-sm font-semibold text-text-primary uppercase tracking-wider flex items-center gap-2">
              <div className="w-6 h-0.5 bg-danger" />
              LightGBM Limitations
            </h3>
            {[
              {
                q: 'Why not just tune LGBM further?',
                a: 'Decision tree splits are axis-aligned — they can only threshold one feature at a time. This misses diagonal/circular decision boundaries inherent in financial factor interactions.',
              },
              {
                q: 'Why can\'t feature engineering fix this?',
                a: 'We tried symbolic regression (GPlearn/PySR) to discover interaction features. But handcrafted features only approximate the interactions — a neural network learns them end-to-end in a continuous, differentiable space.',
              },
              {
                q: 'What about LGBM\'s categorical handling?',
                a: 'LGBM uses optimal histogram splits for categoricals — but this creates discrete bins, not learned latent representations. Entity embeddings map categories into dense vector spaces where semantically similar entities cluster naturally.',
              },
            ].map((item, i) => (
              <div key={i} className="glass-card p-4 hover:border-danger/30 transition-all">
                <div className="flex items-start gap-2 mb-2">
                  <span className="text-danger text-xs font-mono mt-0.5">Q:</span>
                  <h4 className="text-sm font-semibold text-text-primary">{item.q}</h4>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-success text-xs font-mono mt-0.5">A:</span>
                  <p className="text-xs text-text-secondary leading-relaxed">{item.a}</p>
                </div>
              </div>
            ))}
          </div>

          {/* Why MLP specifically */}
          <div className="space-y-3 fade-up opacity-0 stagger-3">
            <h3 className="text-sm font-semibold text-text-primary uppercase tracking-wider flex items-center gap-2">
              <div className="w-6 h-0.5 bg-success" />
              Why MLP + Embeddings?
            </h3>
            {[
              {
                q: 'Why MLP and not CNN/RNN/Transformer?',
                a: 'Our data is cross-sectional tabular, not sequential images or text. Recent research (Gorishniy et al., 2021) shows MLPs with proper regularization match or beat Transformers on tabular data, at 10-50× lower compute cost. We avoid unnecessary complexity.',
              },
              {
                q: 'Why not start with FT-Transformer directly?',
                a: 'Scientific rigor requires progressive complexity. MLP+Embeddings is the simplest deep learning baseline that captures entity representations. It establishes whether learned representations add value before investing in attention mechanisms. We built FT-Transformer as a next step.',
              },
              {
                q: 'Why entity embeddings at all?',
                a: 'Our dataset has high-cardinality categoricals (code, sub_code, sub_category). One-hot encoding would produce 500+ sparse columns. Embeddings compress this into dense ≤50-dim vectors per feature, reducing dimensionality while learning latent structure.',
              },
            ].map((item, i) => (
              <div key={i} className="glass-card p-4 hover:border-success/30 transition-all">
                <div className="flex items-start gap-2 mb-2">
                  <span className="text-danger text-xs font-mono mt-0.5">Q:</span>
                  <h4 className="text-sm font-semibold text-text-primary">{item.q}</h4>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-success text-xs font-mono mt-0.5">A:</span>
                  <p className="text-xs text-text-secondary leading-relaxed">{item.a}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

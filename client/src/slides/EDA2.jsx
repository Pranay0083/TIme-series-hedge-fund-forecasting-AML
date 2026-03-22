import eda2_1 from '../assets/eda2_1.png';
import eda2_2 from '../assets/eda2_2.png';
import eda2_3 from '../assets/eda2_3.png';

export default function EDA2() {
  const findings = [
    {
      title: 'Unseen Test Entities',
      detail: 'Train and test sets share only 12 sub_code entities; 35 test entities are entirely unseen in the training data.',
      icon: '🛑',
      image: eda2_1,
      what: 'Performed strict overlap analysis between train and test categorical sets.',
      why: 'Proved that target-encoding by finite sub_code sets would cause massive leakage and fail structurally on new test data. Forced the model to generalize using macro code groupings instead.'
    },
    {
      title: 'Dummy Noise Injection',
      detail: 'A subset of metrics (feature_b through feature_g) are independent random noise acting as dummy variables.',
      icon: '🗑️',
      image: eda2_2,
      what: 'Ran pairwise correlation, mutual information analysis, and hierarchical clustering on all features.',
      why: 'Permitted explicit removal of noise dummy features to restrict the model\'s capability to artificially overfit on random signals.'
    },
    {
      title: 'Extreme Weight Discrepancies',
      detail: 'Samples contain massive weight discrepancies (spanning 13 orders of magnitude), tightly bundled around the y=0 mark.',
      icon: '⚖️',
      image: eda2_3,
      what: 'Examined sample weights on a logarithmic scale against target distributions.',
      why: 'If used Naively in the loss function, these weights would cause extreme gradient instability. Requires log-transformation or rank-based scaling of weights during training.'
    }
  ]

  return (
    <div className="flex items-center justify-center h-full px-8 pb-10">
      <div className="max-w-6xl w-full">
        <div className="fade-up opacity-0 stagger-1 mb-2">
          <span className="text-xs font-mono text-accent tracking-wider uppercase">04.2 / Exploratory Analysis II</span>
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-6 fade-up opacity-0 stagger-1">
          <span className="gradient-text">Data Leakage Risks & Entity Mismatch</span>
        </h2>

        <div className="grid md:grid-cols-3 gap-6 overflow-y-auto max-h-[70vh] p-2 hide-scrollbar">
          {findings.map((f, i) => (
            <div key={f.title} className={`glass-card p-5 fade-up opacity-0 stagger-${i + 2} hover:border-primary/50 transition-all duration-300 group flex flex-col`}>
              <div className="flex items-center gap-3 mb-3">
                <div className="text-3xl bg-secondary/20 p-2 rounded-lg">{f.icon}</div>
                <h3 className="text-lg font-bold text-text-primary group-hover:text-primary-light transition-colors">{f.title}</h3>
              </div>
              <p className="text-sm border-l-2 border-accent/50 pl-3 mb-4 text-text-secondary leading-relaxed flex-grow">{f.detail}</p>
              
              <div className="mb-4 w-full bg-surface-darker rounded-lg overflow-hidden border border-white/5 h-32 flex items-center justify-center">
                <img src={f.image} alt={f.title} className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" />
              </div>

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
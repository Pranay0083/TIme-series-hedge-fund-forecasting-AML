import eda3_1 from '../assets/eda3_1.png';
import eda3_2 from '../assets/eda3_2.png';
import eda3_3 from '../assets/eda3_3.png';

export default function EDA3() {
  const findings = [
    {
      title: 'Expanding Universe Growth',
      detail: 'The cross-sectional universe size grows systematically over time, making early indexes highly sparse compared to recent data.',
      icon: '🌌',
      image: eda3_1,
      what: 'Plotted universe entity counts sequentially against the discrete ts_index time axis.',
      why: 'Models trained equally on early sparse data might fail to capture the complex cross-sectional interactions present in the dense, modern regime.'
    },
    {
      title: 'Unstable Covariance Structures',
      detail: 'The underlying covariance structure between features is structurally unstable across different market phases.',
      icon: '🌪️',
      image: eda3_2,
      what: 'Tracked the Frobenius norm of feature correlation matrices across distinct temporal rolling windows.',
      why: 'Highlighted the danger of randomized k-fold splits. Dictated the explicit need for an expanding-window chronological validation strategy.'
    },
    {
      title: 'Feature Metric Drift',
      detail: 'ts_index exhibits strong upward/downward drift in mean and standard deviation for several features over time.',
      icon: '📉',
      image: eda3_3,
      what: 'Conducted feature rolling-mean validations and Kolmogorov-Smirnov (KS) tests to quantify train-test distribution shifts.',
      why: 'Motivated recency-weighted modeling (temporal weight decay) to handle rolling structural data regimes and forget outdated patterns.'
    }
  ]

  return (
    <div className="flex items-center justify-center h-full px-8 pb-10">
      <div className="max-w-6xl w-full">
        <div className="fade-up opacity-0 stagger-1 mb-2">
          <span className="text-xs font-mono text-accent tracking-wider uppercase">04.3 / Exploratory Analysis III</span>
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-6 fade-up opacity-0 stagger-1">
          <span className="gradient-text">Temporal Non-Stationarity & Universe Drift</span>
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
export default function EDA1() {
  const findings = [
    {
      title: 'Structural Target Variance',
      detail: 'Target variance strictly scales with the forecasting horizon. Standard deviation increases from 11.7 to 52.8 from horizon 1 to 25.',
      icon: '📈',
      what: 'Aggregated target standard deviation broken down by horizon.',
      why: 'Identified the necessity to normalize the target scaling per horizon. Without this, long-horizon target variances would completely dominate the loss function, ignoring short-term signals.'
    },
    {
      title: 'Extreme Kurtosis (Fat Tails)',
      detail: 'The target variable (y_target) has an extreme kurtosis (~290) featuring a near-zero center and highly fat tails.',
      icon: '📊',
      what: 'Analyzed the target distribution using clipping, histograms, qq-plots, and ran Shapiro-Wilk/Anderson-Darling tests.',
      why: 'Established the fact that standard loss functions (like MSE) would fail completely. We must use robust losses like Huber or MAE to handle these non-Gaussian tails without gradient explosion.'
    },
    {
      title: 'Large-Scale Multi-Horizon Structure',
      detail: 'The dataset is massive (5.3M rows, 94 features) with multiple distinct forecasting horizons (1, 3, 10, 25) embedded within.',
      icon: '🌐',
      what: 'Dataset dimension analysis highlighting the scale and the overlap across temporal forecasting targets.',
      why: 'Dictates the computational strategy and confirms that a single monolithic model might struggle with the distinct dynamics of different horizons.'
    }
  ]

  return (
    <div className="flex items-center justify-center h-full px-8 pb-10">
      <div className="max-w-6xl w-full">
        <div className="fade-up opacity-0 stagger-1 mb-2">
          <span className="text-xs font-mono text-accent tracking-wider uppercase">04.1 / Exploratory Analysis I</span>
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-6 fade-up opacity-0 stagger-1">
          <span className="gradient-text">Data Structure & Target Anomalies</span>
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

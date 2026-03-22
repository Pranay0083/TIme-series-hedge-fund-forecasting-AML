import { useState, useEffect, useCallback } from 'react'
import './index.css'
import TitleSlide from './slides/TitleSlide'
import ProblemSlide from './slides/ProblemSlide'
import DatasetSlide from './slides/DatasetSlide'
import EDA1 from './slides/EDA1'
import EDA2 from './slides/EDA2'
import EDA3 from './slides/EDA3'
import PreprocessSlide from './slides/PreprocessSlide'
import FeatureEng1 from './slides/FeatureEng1'
import FeatureEng2 from './slides/FeatureEng2'
import SymbolicRegressionSlide from './slides/SymbolicRegressionSlide'
import ModelArchSlide from './slides/ModelArchSlide'
import AdvancedSlide from './slides/AdvancedSlide'
import ResultsSlide from './slides/ResultsSlide'
import ICAnalysisSlide from './slides/ICAnalysisSlide'
import ConclusionSlide from './slides/ConclusionSlide'

const slides = [
  { component: TitleSlide, title: 'Title' },
  { component: ProblemSlide, title: 'Problem' },
  { component: DatasetSlide, title: 'Dataset' },
  { component: EDA1, title: 'EDA I - Data & Targets' },
  { component: EDA2, title: 'EDA II - Leakage & Noise' },
  { component: EDA3, title: 'EDA III - Temporal Shifts' },
  { component: PreprocessSlide, title: 'Preprocessing' },
  { component: FeatureEng1, title: 'Features I - Validation' },
  { component: FeatureEng2, title: 'Features II - TSFresh' },
  { component: SymbolicRegressionSlide, title: 'Symbolic Regression' },
  { component: ModelArchSlide, title: 'Model' },
  { component: AdvancedSlide, title: 'Advanced' },
  { component: ICAnalysisSlide, title: 'IC Analysis' },
  { component: ResultsSlide, title: 'Results' },
  { component: ConclusionSlide, title: 'Conclusion' },
]

function App() {
  const [current, setCurrent] = useState(0)
  const [direction, setDirection] = useState(1)
  const [isTransitioning, setIsTransitioning] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const goTo = useCallback((index) => {
    if (index === current || isTransitioning || index < 0 || index >= slides.length) return
    setDirection(index > current ? 1 : -1)
    setIsTransitioning(true)
    setTimeout(() => {
      setCurrent(index)
      setIsTransitioning(false)
    }, 350)
  }, [current, isTransitioning])

  const next = useCallback(() => goTo(current + 1), [current, goTo])
  const prev = useCallback(() => goTo(current - 1), [current, goTo])

  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown' || e.key === ' ') {
        e.preventDefault()
        next()
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault()
        prev()
      } else if (e.key === 'Escape') {
        setSidebarOpen(s => !s)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [next, prev])

  const SlideComponent = slides[current].component

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-surface noise-bg">
      {/* Ambient background orbs */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -left-40 w-96 h-96 rounded-full opacity-20"
          style={{ background: 'radial-gradient(circle, rgba(99,102,241,0.4) 0%, transparent 70%)', animation: 'float 8s ease-in-out infinite' }} />
        <div className="absolute -bottom-40 -right-40 w-[500px] h-[500px] rounded-full opacity-15"
          style={{ background: 'radial-gradient(circle, rgba(34,211,238,0.3) 0%, transparent 70%)', animation: 'float 10s ease-in-out infinite reverse' }} />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] rounded-full opacity-5"
          style={{ background: 'radial-gradient(circle, rgba(99,102,241,0.3) 0%, transparent 60%)' }} />
      </div>

      {/* Sidebar navigation */}
      <div className={`fixed top-0 left-0 h-full w-64 z-50 glass transition-transform duration-300 ease-out ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="p-6">
          <h3 className="text-sm font-semibold text-text-muted uppercase tracking-wider mb-6">Slides</h3>
          <div className="space-y-1">
            {slides.map((slide, i) => (
              <button key={i} onClick={() => { goTo(i); setSidebarOpen(false) }}
                className={`w-full text-left px-4 py-2.5 rounded-lg text-sm transition-all duration-200 ${
                  i === current 
                    ? 'bg-primary/20 text-primary-light font-medium' 
                    : 'text-text-secondary hover:text-text-primary hover:bg-surface-lighter'
                }`}>
                <span className="text-text-muted mr-3 font-mono text-xs">{String(i + 1).padStart(2, '0')}</span>
                {slide.title}
              </button>
            ))}
          </div>
        </div>
      </div>
      {sidebarOpen && <div className="fixed inset-0 z-40 bg-black/40" onClick={() => setSidebarOpen(false)} />}

      {/* Top bar */}
      <header className="fixed top-0 left-0 right-0 z-30 px-6 py-4 flex items-center justify-between">
        <button onClick={() => setSidebarOpen(s => !s)}
          className="flex items-center gap-2 text-text-muted hover:text-text-primary transition-colors group">
          <div className="flex flex-col gap-1">
            <span className="block w-5 h-0.5 bg-text-muted group-hover:bg-primary-light transition-all group-hover:w-6" />
            <span className="block w-4 h-0.5 bg-text-muted group-hover:bg-primary-light transition-all group-hover:w-6" />
            <span className="block w-3 h-0.5 bg-text-muted group-hover:bg-primary-light transition-all group-hover:w-6" />
          </div>
        </button>
        <div className="flex items-center gap-3">
          <span className="text-xs font-mono text-text-muted">
            {String(current + 1).padStart(2, '0')} / {String(slides.length).padStart(2, '0')}
          </span>
          <div className="flex gap-1">
            {slides.map((_, i) => (
              <button key={i} onClick={() => goTo(i)}
                className={`h-1 rounded-full transition-all duration-300 ${
                  i === current ? 'w-6 bg-primary' : i < current ? 'w-2 bg-primary/40' : 'w-2 bg-border'
                }`} />
            ))}
          </div>
        </div>
      </header>

      {/* Slide content */}
      <main className="relative z-10 w-full h-full">
        <div key={current}
          className={`w-full h-full ${isTransitioning ? 'slide-exit' : 'slide-enter'}`}>
          <SlideComponent />
        </div>
      </main>

      {/* Bottom navigation */}
      <footer className="fixed bottom-0 left-0 right-0 z-30 px-8 py-5 flex items-center justify-between">
        <button onClick={prev} disabled={current === 0}
          className="flex items-center gap-2 text-sm text-text-muted hover:text-text-primary disabled:opacity-20 disabled:cursor-not-allowed transition-all group">
          <svg className="w-4 h-4 transition-transform group-hover:-translate-x-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Previous
        </button>
        <span className="text-xs text-text-muted font-mono hidden sm:block">
          Press <kbd className="px-1.5 py-0.5 mx-0.5 rounded bg-surface-lighter text-text-secondary border border-border text-[10px]">Space</kbd> or arrows to navigate
        </span>
        <button onClick={next} disabled={current === slides.length - 1}
          className="flex items-center gap-2 text-sm text-text-muted hover:text-text-primary disabled:opacity-20 disabled:cursor-not-allowed transition-all group">
          Next
          <svg className="w-4 h-4 transition-transform group-hover:translate-x-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </button>
      </footer>
    </div>
  )
}

export default App

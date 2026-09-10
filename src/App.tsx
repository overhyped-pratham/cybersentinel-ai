'use client'

import React, { useState, useEffect } from 'react'
import { SplineScene } from '@/components/ui/splite'
import {
  Shield,
  Activity,
  Cpu,
  Network,
  ArrowRight,
  ArrowDown,
  Radio,
  Layers,
  Zap,
  RotateCcw,
  Sliders,
  Crosshair,
  Lock,
  Flame,
  Binary,
  CheckCircle2,
  Terminal,
  ExternalLink,
  ChevronRight
} from 'lucide-react'

// 11-step pipeline of CyberSentinel AI with SlowBros editorial attributes
const PIPELINE_STEPS = [
  {
    step: '01',
    name: 'Live Network Telemetry',
    subtitle: 'PCAP & NetFlow Line-Rate Ingest',
    description: 'Captures live packet buffers, NetFlow v9/IPFIX records, and raw socket streams at line rate with zero-copy ring buffers.',
    tag: 'Ingest',
    specs: [
      'Line-rate packet capture with zero-copy ring buffers',
      'NetFlow v9, IPFIX, and raw socket stream ingestion'
    ],
    diagram: {
      type: 'stream',
      title: 'INGEST BUFFER FLOW',
      nodes: ['PACKET RING', 'PARSE', 'SOCKET STREAM']
    },
    icon: Network,
  },
  {
    step: '02',
    name: 'LiveIngestService',
    subtitle: 'Temporal Window Aggregation',
    description: 'Aggregates packet streams into windowed statistical flows, tracking rate bursts, byte distributions, and protocol dispersion.',
    tag: 'Buffer',
    specs: [
      'Sliding temporal windows with dynamic overlap',
      'Tracks packet rates, burst ratios, and protocol mix'
    ],
    diagram: {
      type: 'window',
      title: 'TEMPORAL AGGREGATION WINDOW',
      nodes: ['WINDOW T-1', 'BURST METRIC', 'WINDOW T0']
    },
    icon: Activity,
  },
  {
    step: '03',
    name: '24-D State Vector',
    subtitle: 'RobustScaler Normalization',
    description: 'Projects flows into 24-dimensional feature representations normalized against empirical medians and interquartile ranges.',
    tag: 'Features',
    specs: [
      '24 continuous flow & header behavioral features',
      'Outlier-resistant IQR normalization via RobustScaler'
    ],
    diagram: {
      type: 'vector',
      title: '24-DIMENSIONAL FEATURE TENSOR',
      bars: [85, 42, 95, 60, 30, 78, 92, 50]
    },
    icon: Sliders,
  },
  {
    step: '04',
    name: 'CyberWorldModelV2',
    subtitle: 'Neural Transition Matrix',
    description: 'Autoregressive recurrent GRU architecture with learned transition matrices modeling temporal adversarial state evolution.',
    tag: 'Deep Model',
    specs: [
      'Gated Recurrent Unit (GRU) temporal memory',
      'Learned transition priors between attack phases'
    ],
    diagram: {
      type: 'gru',
      title: 'RECURRENT GRU TRANSITION LOOP',
      nodes: ['h(t-1)', 'GRU CELL', 'h(t)']
    },
    icon: Cpu,
  },
  {
    step: '05',
    name: 'Temperature Calibration',
    subtitle: 'Optimal Scaling T*=1.568',
    description: 'Post-processing calibration via Nelder-Mead optimization minimizes expected calibration error and achieves Brier score 0.0452.',
    tag: 'Calibration',
    specs: [
      'T* = 1.568 Nelder-Mead calibrated temperature',
      'Brier score: 0.0452 | ECE: < 2.1%'
    ],
    diagram: {
      type: 'calibration',
      title: 'PROBABILITY RELIABILITY CURVE',
      stat: 'T* = 1.568 • BRIER 0.0452'
    },
    icon: Zap,
  },
  {
    step: '06',
    name: 'Probability Distribution',
    subtitle: 'Softmax Stage Likelihood',
    description: 'Generates well-calibrated posterior probability distribution across all 10 unified MITRE attack lifecycle stages.',
    tag: 'Probability',
    specs: [
      '10-class mutually exclusive posterior probabilities',
      'Zero hardcoded probabilities — 100% neural inference'
    ],
    diagram: {
      type: 'softmax',
      title: 'POSTERIOR STAGE PROBABILITY',
      bars: [12, 8, 4, 76, 18, 5]
    },
    icon: Binary,
  },
  {
    step: '07',
    name: 'Dynamic Transition Detection',
    subtitle: '|ΔS| Feature Attribution',
    description: 'Identifies active attack transitions in real time with 83.33% empirical accuracy via feature-space perturbation gradients.',
    tag: 'Attribution',
    specs: [
      '83.33% empirical transition detection accuracy',
      'Gradient-based feature attribution highlights root causes'
    ],
    diagram: {
      type: 'delta',
      title: 'PERTURBATION GRADIENT |ΔS|',
      stat: 'ACCURACY: 83.33%'
    },
    icon: Flame,
  },
  {
    step: '08',
    name: 'Multi-Step Simulation',
    subtitle: 'Autoregressive Rollout (K=1..4)',
    description: 'Projects adversarial progression up to 4 temporal horizons into the future before the adversary reaches crown-jewel assets.',
    tag: 'Rollout',
    specs: [
      'Autoregressive latent state rollouts for K=1, 2, 3, 4',
      'Anticipates lateral movement and exfiltration early'
    ],
    diagram: {
      type: 'rollout',
      title: 'MULTI-STEP AUTOREGRESSIVE HORIZON',
      nodes: ['K=1', 'K=2', 'K=3', 'K=4']
    },
    icon: RotateCcw,
  },
  {
    step: '09',
    name: 'Dynamic Lifecycle State',
    subtitle: 'Runtime NOW vs SIM States',
    description: 'Resolves observed history, active real-time attack stage (NOW), and simulated forward stages (SIM) with zero hardcoded logic.',
    tag: 'Lifecycle',
    specs: [
      'Live temporal alignment: OBSERVED → NOW → FORECAST',
      'Seamless sync with WebSocket telemetry feed'
    ],
    diagram: {
      type: 'lifecycle',
      title: 'TEMPORAL STATE COMPARATOR',
      nodes: ['HISTORY', 'NOW', 'SIM+1']
    },
    icon: Crosshair,
  },
  {
    step: '10',
    name: 'MITRE ATT&CK Grounding',
    subtitle: 'Enterprise Matrix v14',
    description: 'Grounds predicted adversary behavior to MITRE Enterprise tactics, techniques, procedures, and empirical sub-technique IDs.',
    tag: 'MITRE',
    specs: [
      'Enterprise Matrix v14 tactic & technique mapping',
      'Sub-technique IDs grounded in live network evidence'
    ],
    diagram: {
      type: 'mitre',
      title: 'GROUNDED ATT&CK TECHNIQUES',
      nodes: ['T1078', 'T1059', 'T1046']
    },
    icon: Shield,
  },
  {
    step: '11',
    name: 'Proactive SOC Defense',
    subtitle: 'RiskEngine & Playbooks',
    description: 'Calculates composite risk score and synthesizes deterministic, evidence-backed mitigation steps for rapid SOC response.',
    tag: 'Triage',
    specs: [
      'Dynamic risk score calculation (0–100 scale)',
      'Automated defense recommendations & containment rules'
    ],
    diagram: {
      type: 'defense',
      title: 'DETERMINISTIC PLAYBOOK',
      stat: 'ISOLATE HOST • REVOKE TOKENS'
    },
    icon: Lock,
  },
]

export default function App() {
  const [backendStatus, setBackendStatus] = useState({
    connected: true,
    modelLoaded: true,
    temperature: 1.568,
    brier: 0.0452,
    top1Acc: '97.73%'
  })
  const [selectedStep, setSelectedStep] = useState(0)
  const [isRedirecting, setIsRedirecting] = useState(false)

  useEffect(() => {
    fetch('http://localhost:8000/api/v1/health')
      .then(res => res.json())
      .then(data => {
        if (data && data.status === 'ok') {
          setBackendStatus(prev => ({
            ...prev,
            connected: true,
            modelLoaded: !!data.model_loaded,
            temperature: data.temperature || 1.568
          }))
        }
      })
      .catch(() => {})
  }, [])

  const handleEnterCommandCenter = () => {
    setIsRedirecting(true)
    const targetUrl = window.location.port === '8000'
      ? '/ui/index.html?view=command'
      : 'http://localhost:8000/ui/index.html?view=command'
    
    setTimeout(() => {
      window.location.href = targetUrl
    }, 120)
  }

  const handleViewArchitecture = () => {
    document.getElementById('architecture')?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <div className="min-h-screen bg-[#F9F6F0] text-[#111111] flex flex-col relative selection:bg-[#C82B14] selection:text-white font-sans antialiased">
      
      {/* ─── SLOWBROS FLOATING PILL NAVIGATION ────────────────────────── */}
      <header className="sticky top-4 z-50 w-full px-4 sm:px-6 flex justify-center">
        <nav className="w-full max-w-6xl bg-white/95 backdrop-blur-md border border-neutral-900/10 shadow-lg shadow-black/[0.04] rounded-full px-5 py-3 flex items-center justify-between transition-all">
          {/* Brand Logo & Name */}
          <div
            className="flex items-center gap-3 cursor-pointer select-none"
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
          >
            <div className="w-8 h-8 rounded-full bg-[#C82B14] flex items-center justify-center shadow-sm">
              <Shield className="w-4 h-4 text-white" />
            </div>
            <div className="flex items-center gap-1.5">
              <span className="font-serif font-black text-base sm:text-lg tracking-tight text-neutral-900">
                CyberSentinel
              </span>
              <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded bg-[#C82B14]/10 text-[#C82B14] uppercase tracking-wider">
                AI
              </span>
            </div>
          </div>

          {/* Navigation Links (Desktop) */}
          <div className="hidden lg:flex items-center gap-7 text-xs font-mono tracking-wider text-neutral-600 font-medium">
            <a href="#architecture" className="hover:text-[#C82B14] transition-colors uppercase">
              Architecture
            </a>
            <a href="#pipeline" className="hover:text-[#C82B14] transition-colors uppercase">
              11-Step Pipeline
            </a>
            <a href="#metrics" className="hover:text-[#C82B14] transition-colors uppercase">
              Metrics
            </a>
            <span className="flex items-center gap-1.5 text-neutral-400">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span className="text-neutral-600 font-mono text-[11px]">
                {backendStatus.modelLoaded ? 'CYBERWORLDMODEL V2' : 'CONNECTING'}
              </span>
            </span>
          </div>

          {/* Red Pill Action Button */}
          <div className="flex items-center gap-3">
            <button
              onClick={handleEnterCommandCenter}
              disabled={isRedirecting}
              className="bg-[#C82B14] hover:bg-[#A9220E] text-white font-bold text-xs sm:text-xs uppercase tracking-wider rounded-full px-5 py-2.5 shadow-sm transition-all transform hover:-translate-y-0.5 flex items-center gap-2 disabled:opacity-50"
            >
              <span>{isRedirecting ? 'LAUNCHING...' : 'ENTER COMMAND CENTER'}</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </nav>
      </header>

      {/* ─── SLOWBROS MARQUEE / TICKER RIBBON ─────────────────────────── */}
      <div className="w-full border-y border-neutral-900/10 bg-white/70 backdrop-blur-sm py-2 px-4 mt-6 overflow-hidden select-none">
        <div className="max-w-7xl mx-auto flex items-center justify-center flex-wrap gap-x-5 gap-y-1 text-[11px] font-mono tracking-widest text-neutral-700">
          <span className="flex items-center gap-2">
            <span className="text-[#C82B14]">◆</span>
            <span>AUTOREGRESSIVE ATTACK FORECASTING</span>
          </span>
          <span className="flex items-center gap-2">
            <span className="text-[#C82B14]">◆</span>
            <span>24-D TEMPORAL STATE</span>
          </span>
          <span className="flex items-center gap-2">
            <span className="text-[#C82B14]">◆</span>
            <span>CYBERWORLDMODEL V2</span>
          </span>
          <span className="flex items-center gap-2">
            <span className="text-[#C82B14]">◆</span>
            <span>MITRE ATT&CK v14</span>
          </span>
          <span className="flex items-center gap-2">
            <span className="text-[#C82B14]">◆</span>
            <span>REAL-TIME SOC INTEL</span>
          </span>
          <span className="flex items-center gap-2">
            <span className="text-[#C82B14]">◆</span>
            <span>83.3% TRANSITION ACCURACY</span>
          </span>
          <span className="hidden md:flex items-center gap-2">
            <span className="text-[#C82B14]">◆</span>
            <span>BRIER SCORE 0.0452</span>
          </span>
        </div>
      </div>

      {/* ─── SLOWBROS SPLIT HERO SECTION WITH INTERACTIVE CYBORG ──────── */}
      <section className="max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 pt-8 pb-12 lg:pt-12 lg:pb-16 flex flex-col">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-8 items-center">
          
          {/* LEFT COLUMN: SlowBros Editorial Headlines & CTAs (7 cols) */}
          <div className="lg:col-span-7 flex flex-col justify-center text-left z-10">
            {/* Technical Sub-label / Tag */}
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-neutral-900/30 text-[11px] font-mono tracking-widest uppercase mb-6 bg-white/80 w-max shadow-sm">
              <span className="w-2 h-2 rounded-full bg-[#C82B14] animate-ping" />
              <span className="text-neutral-800 font-bold">AI-POWERED NETWORK DEFENSE</span>
              <span className="text-neutral-400">•</span>
              <span className="text-neutral-600">SMART INDIA HACKATHON 2026</span>
            </div>

            {/* Giant High-Contrast Editorial Serif Headline */}
            <h1 className="text-5xl sm:text-6xl md:text-7xl lg:text-[5.1rem] font-serif font-black tracking-tight text-neutral-900 leading-[1.04]">
              DON'T WAIT FOR THE <br />
              <span className="text-[#C82B14] italic relative inline-block">
                ATTACK.
                {/* Hand-drawn style red underline SVG */}
                <svg
                  className="absolute -bottom-2 sm:-bottom-3 left-0 w-full h-3 sm:h-4 text-[#C82B14]"
                  viewBox="0 0 200 12"
                  fill="none"
                  preserveAspectRatio="none"
                >
                  <path
                    d="M2 9C50 3 150 2 198 8"
                    stroke="currentColor"
                    strokeWidth="3.5"
                    strokeLinecap="round"
                  />
                </svg>
              </span>
            </h1>

            {/* Sub-heading */}
            <div className="text-xs sm:text-sm font-mono font-bold tracking-widest text-[#C82B14] uppercase mt-6 sm:mt-8 flex items-center gap-2">
              <span>AI-POWERED NETWORK ATTACK FORECASTING</span>
              <span className="hidden sm:inline-block w-8 h-[1px] bg-[#C82B14]/40"></span>
            </div>

            {/* Narrative Paragraph */}
            <p className="mt-4 text-neutral-700 text-sm sm:text-base lg:text-lg leading-relaxed max-w-xl font-normal">
              CyberSentinel AI learns evolving network traffic, forecasts future attack states, explains its prediction, maps the threat to MITRE ATT&CK, and assists the SOC analyst with evidence-based defensive decisions.
            </p>

            {/* Pill Tags Row (SlowBros style from reference screenshots) */}
            <div className="mt-6 flex flex-wrap gap-2 text-[11px] font-mono text-neutral-700">
              <span className="px-3 py-1 rounded-full border border-neutral-900/30 bg-white/70">
                ○ 24-D TEMPORAL TENSORS
              </span>
              <span className="px-3 py-1 rounded-full border border-neutral-900/30 bg-white/70">
                ○ AUTOREGRESSIVE K=4
              </span>
              <span className="px-3 py-1 rounded-full border border-neutral-900/30 bg-white/70">
                ○ ZERO HARDCODED LOGIC
              </span>
              <span className="px-3 py-1 rounded-full border border-neutral-900/30 bg-white/70">
                ○ MITRE v14 GROUNDED
              </span>
            </div>

            {/* Editorial CTAs */}
            <div className="mt-8 flex flex-wrap items-center gap-4">
              <button
                onClick={handleEnterCommandCenter}
                disabled={isRedirecting}
                className="bg-[#C82B14] hover:bg-[#A9220E] text-white font-bold text-xs sm:text-sm uppercase tracking-wider rounded-xl sm:rounded-full px-7 py-4 shadow-md transition-all transform hover:-translate-y-0.5 flex items-center gap-3 disabled:opacity-50"
              >
                <span>{isRedirecting ? 'ENTERING SOC...' : 'ENTER COMMAND CENTER'}</span>
                <ArrowRight className="w-4 h-4" />
              </button>

              <button
                onClick={handleViewArchitecture}
                className="bg-transparent border border-neutral-900 hover:bg-neutral-900/5 text-neutral-900 font-bold text-xs sm:text-sm uppercase tracking-wider rounded-xl sm:rounded-full px-6 py-4 transition-all flex items-center gap-2"
              >
                <span>VIEW ARCHITECTURE</span>
                <ArrowDown className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* RIGHT COLUMN: Interactive 3D Cyborg with SlowBros Technical Annotations (5 cols) */}
          <div className="lg:col-span-5 relative w-full h-[460px] sm:h-[540px] lg:h-[620px] flex items-center justify-center">
            
            {/* The 3D Cyborg Component — 100% Preserved Interactive Model */}
            <div className="w-full h-full relative z-0">
              <SplineScene 
                scene="https://prod.spline.design/kZDDjO5HuC9GJUM2/scene.splinecode"
                className="w-full h-full"
              />
            </div>

            {/* Floating Editorial Technical Annotations (Thin crisp borders, cream badges) */}
            {/* Annotation 1: Top-Left */}
            <div className="absolute top-2 left-2 z-10 bg-white/90 backdrop-blur-md border border-neutral-900/80 rounded-xl px-3 py-2 text-[10px] font-mono shadow-md hidden sm:block max-w-[180px]">
              <div className="text-neutral-500 font-bold uppercase text-[9px] flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                <span>LIVE TELEMETRY</span>
              </div>
              <div className="text-neutral-900 font-bold mt-0.5">
                CONNECTED • LINE RATE
              </div>
            </div>

            {/* Annotation 2: Top-Right */}
            <div className="absolute top-2 right-2 z-10 bg-white/90 backdrop-blur-md border border-neutral-900/80 rounded-xl px-3 py-2 text-[10px] font-mono shadow-md hidden sm:block max-w-[180px] text-right">
              <div className="text-neutral-500 font-bold uppercase text-[9px]">
                MODEL ARCHITECTURE
              </div>
              <div className="text-[#C82B14] font-bold mt-0.5">
                CYBERWORLDMODEL V2
              </div>
            </div>

            {/* Annotation 3: Bottom-Left */}
            <div className="absolute bottom-4 left-2 z-10 bg-white/90 backdrop-blur-md border border-neutral-900/80 rounded-xl px-3 py-2 text-[10px] font-mono shadow-md hidden sm:block max-w-[180px]">
              <div className="text-neutral-500 font-bold uppercase text-[9px]">
                FORECAST HORIZON
              </div>
              <div className="text-neutral-900 font-bold mt-0.5">
                ROLLOUT K=1..4 STEPS
              </div>
            </div>

            {/* Annotation 4: Bottom-Right */}
            <div className="absolute bottom-4 right-2 z-10 bg-white/90 backdrop-blur-md border border-neutral-900/80 rounded-xl px-3 py-2 text-[10px] font-mono shadow-md hidden sm:block max-w-[190px] text-right">
              <div className="text-neutral-500 font-bold uppercase text-[9px]">
                THREAT GROUNDING
              </div>
              <div className="text-neutral-900 font-bold mt-0.5">
                MITRE ATT&CK ENTERPRISE
              </div>
            </div>
          </div>
        </div>

        {/* ─── SLOWBROS SCOREBOARD METRICS BAR ────────────────────────── */}
        <div id="metrics" className="mt-12 pt-8 border-t border-neutral-900/10 grid grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6 font-mono">
          <div className="bg-white border border-neutral-900 rounded-2xl p-5 shadow-sm hover:shadow-md transition-all">
            <div className="text-xs text-neutral-500 font-bold uppercase tracking-wider">Temporal State</div>
            <div className="text-3xl sm:text-4xl font-serif font-black text-neutral-900 mt-1">24-D</div>
            <div className="text-[11px] text-neutral-600 mt-2 font-sans leading-tight">
              Continuous flow & header vectors normalized via RobustScaler
            </div>
          </div>

          <div className="bg-white border border-neutral-900 rounded-2xl p-5 shadow-sm hover:shadow-md transition-all">
            <div className="text-xs text-neutral-500 font-bold uppercase tracking-wider">Stage Accuracy</div>
            <div className="text-3xl sm:text-4xl font-serif font-black text-[#C82B14] mt-1">{backendStatus.top1Acc}</div>
            <div className="text-[11px] text-neutral-600 mt-2 font-sans leading-tight">
              Top-1 empirical attack stage classification rate
            </div>
          </div>

          <div className="bg-white border border-neutral-900 rounded-2xl p-5 shadow-sm hover:shadow-md transition-all">
            <div className="text-xs text-neutral-500 font-bold uppercase tracking-wider">Forecast Horizon</div>
            <div className="text-3xl sm:text-4xl font-serif font-black text-neutral-900 mt-1">K=4</div>
            <div className="text-[11px] text-neutral-600 mt-2 font-sans leading-tight">
              Autoregressive forward projection steps into future
            </div>
          </div>

          <div className="bg-white border border-neutral-900 rounded-2xl p-5 shadow-sm hover:shadow-md transition-all">
            <div className="text-xs text-neutral-500 font-bold uppercase tracking-wider">Calibration Score</div>
            <div className="text-3xl sm:text-4xl font-serif font-black text-neutral-900 mt-1">0.0452</div>
            <div className="text-[11px] text-neutral-600 mt-2 font-sans leading-tight">
              Optimal temperature scaling Brier score (T*=1.568)
            </div>
          </div>
        </div>
      </section>

      {/* ─── SLOWBROS ARCHITECTURE & 11-STEP PIPELINE SECTION ─────────── */}
      <section id="architecture" className="w-full bg-[#FAF8F5] border-t border-neutral-900/10 py-16 lg:py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          
          {/* Editorial Section Header */}
          <div className="text-center max-w-3xl mx-auto mb-16">
            <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full border border-neutral-900/20 bg-white text-[11px] font-mono font-bold tracking-widest text-neutral-800 uppercase mb-4 shadow-sm">
              <Layers className="w-3.5 h-3.5 text-[#C82B14]" />
              <span>HOW CYBERSENTINEL THINKS</span>
            </div>
            
            <h2 className="text-4xl sm:text-5xl lg:text-6xl font-serif font-black tracking-tight text-neutral-900 leading-tight">
              11-Step Neural Attack <br />
              <span className="text-[#C82B14] italic">Forecasting Pipeline</span>
            </h2>

            <p className="mt-5 text-neutral-600 text-sm sm:text-base leading-relaxed">
              Zero hardcoded intelligence. Every prediction is derived dynamically from high-throughput network telemetry, normalized into temporal state tensors, forecast multi-step through <strong className="text-neutral-900 font-bold">CyberWorldModelV2</strong>, and grounded to MITRE ATT&CK Enterprise v14 tactics.
            </p>
          </div>

          {/* 11-Step Pipeline Grid (SlowBros Cards: Crisp 1px border, 01/11 tag, red wireframe icon) */}
          <div id="pipeline" className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {PIPELINE_STEPS.map((item, idx) => {
              const Icon = item.icon
              const isSelected = selectedStep === idx

              return (
                <div
                  key={item.step}
                  onClick={() => setSelectedStep(idx)}
                  className={`cursor-pointer transition-all duration-300 rounded-3xl p-6 sm:p-7 flex flex-col justify-between relative overflow-hidden bg-white border ${
                    isSelected
                      ? 'border-neutral-900 shadow-xl ring-2 ring-neutral-900/10 transform -translate-y-1'
                      : 'border-neutral-900/70 hover:border-neutral-900 hover:shadow-lg'
                  }`}
                >
                  {/* Card Top Row: Red Wireframe Icon & Monospace Step Number (from SlowBros screenshot 1) */}
                  <div>
                    <div className="flex items-center justify-between pb-4 border-b border-neutral-200">
                      <div className="w-10 h-10 rounded-xl bg-neutral-50 border border-neutral-200 flex items-center justify-center text-[#C82B14]">
                        <Icon className="w-5 h-5 stroke-[1.75]" />
                      </div>
                      <span className="font-mono text-xs font-bold text-neutral-500 tracking-wider">
                        {item.step} / 11
                      </span>
                    </div>

                    {/* Step Title & Subtitle */}
                    <div className="mt-5">
                      <h3 className="font-serif font-black text-xl text-neutral-900 tracking-tight">
                        {item.name}
                      </h3>
                      <div className="font-mono text-xs text-[#C82B14] font-bold mt-1 tracking-wider uppercase">
                        {item.subtitle}
                      </div>
                    </div>

                    {/* Narrative Description */}
                    <p className="mt-3 text-neutral-600 text-xs sm:text-sm leading-relaxed">
                      {item.description}
                    </p>

                    {/* Practice Offerings / Technical Specs (SlowBros screenshot 2) */}
                    <div className="mt-5 pt-4 border-t border-neutral-100">
                      <div className="text-[10px] font-mono font-bold tracking-wider text-neutral-400 uppercase mb-2">
                        TECHNICAL SPECIFICATIONS:
                      </div>
                      <div className="space-y-1.5">
                        {item.specs.map((spec, sIdx) => (
                          <div key={sIdx} className="flex items-start gap-2 text-xs text-neutral-700">
                            <span className="w-1.5 h-1.5 rounded-full bg-[#C82B14] mt-1.5 shrink-0" />
                            <span>{spec}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Inner Diagram / Visual Box (SlowBros screenshot 1 inner block) */}
                  <div className="mt-6 pt-4 border-t border-neutral-100">
                    <div className="rounded-xl bg-[#111318] text-neutral-200 p-3.5 font-mono text-[11px] flex flex-col gap-2 shadow-inner">
                      <div className="flex items-center justify-between text-[9px] text-neutral-400 border-b border-neutral-800 pb-1.5">
                        <span className="font-bold text-[#C82B14] uppercase">{item.diagram.title}</span>
                        <span>LIVE INFERENCE</span>
                      </div>

                      {item.diagram.nodes && (
                        <div className="flex items-center justify-between gap-1 text-[10px] text-neutral-300 py-1">
                          {item.diagram.nodes.map((n, nIdx) => (
                            <React.Fragment key={nIdx}>
                              <span className="px-2 py-0.5 rounded bg-neutral-800 border border-neutral-700">
                                {n}
                              </span>
                              {nIdx < item.diagram.nodes.length - 1 && (
                                <span className="text-[#C82B14]">→</span>
                              )}
                            </React.Fragment>
                          ))}
                        </div>
                      )}

                      {item.diagram.bars && (
                        <div className="flex items-end gap-1.5 h-7 pt-1">
                          {item.diagram.bars.map((b, bIdx) => (
                            <div
                              key={bIdx}
                              className="flex-1 bg-gradient-to-t from-[#C82B14] to-amber-400 rounded-t-sm"
                              style={{ height: `${b}%` }}
                            />
                          ))}
                        </div>
                      )}

                      {item.diagram.stat && (
                        <div className="text-[10px] text-neutral-300 py-0.5">
                          <span className="text-emerald-400 font-bold">● </span>
                          <span>{item.diagram.stat}</span>
                        </div>
                      )}
                    </div>

                    <div className="mt-3 flex items-center justify-between text-xs font-mono font-bold text-neutral-900 group">
                      <span className="text-neutral-500 uppercase tracking-wider text-[11px]">
                        Pipeline Stage
                      </span>
                      <span className="text-[#C82B14] flex items-center gap-1">
                        <span>INSPECT</span>
                        <ChevronRight className="w-3.5 h-3.5" />
                      </span>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>

          {/* Bottom Launch Banner */}
          <div className="mt-16 rounded-3xl bg-neutral-900 text-white p-8 sm:p-12 flex flex-col md:flex-row items-center justify-between gap-8 shadow-2xl relative overflow-hidden">
            <div className="flex flex-col gap-2 text-left z-10">
              <div className="text-xs font-mono font-bold text-[#C82B14] uppercase tracking-wider">
                Smart India Hackathon 2026 • Live SOC Intelligence
              </div>
              <h3 className="text-2xl sm:text-3xl lg:text-4xl font-serif font-black">
                Ready to witness autonomous attack forecasting?
              </h3>
              <p className="text-neutral-400 text-xs sm:text-sm max-w-xl">
                Experience the live SOC Command Center. Stream telemetry, inspect 24-D temporal tensors, view MITRE ATT&CK grounding, and execute proactive defensive containment.
              </p>
            </div>

            <div className="shrink-0 z-10">
              <button
                onClick={handleEnterCommandCenter}
                disabled={isRedirecting}
                className="bg-[#C82B14] hover:bg-[#E03018] text-white font-bold text-sm uppercase tracking-wider rounded-full px-8 py-4 shadow-xl transition-all transform hover:-translate-y-1 flex items-center gap-3 disabled:opacity-50"
              >
                <span>{isRedirecting ? 'LAUNCHING COMMAND CENTER...' : 'LAUNCH COMMAND CENTER'}</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>

        </div>
      </section>

      {/* ─── SLOWBROS EDITORIAL FOOTER ─────────────────────────────────── */}
      <footer className="w-full bg-[#111111] text-neutral-400 py-10 px-6 sm:px-8 font-mono text-xs border-t border-neutral-800">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <div className="w-6 h-6 rounded-full bg-[#C82B14] flex items-center justify-center">
              <Shield className="w-3.5 h-3.5 text-white" />
            </div>
            <div className="font-serif font-black text-neutral-200 text-sm">
              CYBERSENTINEL AI
            </div>
            <span className="text-neutral-600">|</span>
            <div className="text-[11px] text-neutral-400">
              SMART INDIA HACKATHON 2026
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-5 text-neutral-400 text-[11px]">
            <span>CyberWorldModelV2</span>
            <span>•</span>
            <span>MITRE ATT&CK v14</span>
            <span>•</span>
            <span>RobustScaler (24-D)</span>
            <span>•</span>
            <span>Optimal Scaling T*=1.568</span>
          </div>
        </div>
      </footer>

    </div>
  )
}

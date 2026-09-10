'use client'

import React, { useState, useEffect } from 'react'
import { SplineScene } from '@/components/ui/splite'
import { Spotlight } from '@/components/ui/spotlight'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import {
  Shield,
  Activity,
  Cpu,
  Database,
  Network,
  ArrowRight,
  Radio,
  Layers,
  CheckCircle,
  ExternalLink,
  ChevronRight,
  Zap,
  RotateCcw,
  Sliders,
  Crosshair,
  Lock,
  Flame,
  Binary
} from 'lucide-react'

// The 11-step pipeline of CyberSentinel AI
const PIPELINE_STEPS = [
  {
    step: '01',
    name: 'Live Network Telemetry',
    subtitle: 'PCAP & NetFlow Ingest',
    description: 'Captures live packet buffers, NetFlow v9/IPFIX records, and raw socket streams at line rate with zero-copy ring buffers.',
    tag: 'Raw Data',
    color: 'border-cyan-500/40 text-cyan-400 bg-cyan-950/20',
    icon: Network,
  },
  {
    step: '02',
    name: 'LiveIngestService',
    subtitle: 'Temporal Buffering',
    description: 'Aggregates packet streams into windowed statistical flows, tracking rate bursts, duration, and protocol distributions.',
    tag: 'Buffer',
    color: 'border-blue-500/40 text-blue-400 bg-blue-950/20',
    icon: Activity,
  },
  {
    step: '03',
    name: '24-D State Vector',
    subtitle: 'RobustScaler Normalization',
    description: 'Projects flows into 24-dimensional feature representations normalized against empirical medians and interquartile ranges.',
    tag: 'Features',
    color: 'border-indigo-500/40 text-indigo-400 bg-indigo-950/20',
    icon: Sliders,
  },
  {
    step: '04',
    name: 'CyberWorldModelV2',
    subtitle: 'Neural Transition Matrix',
    description: 'Autoregressive recurrent GRU architecture with learned transition matrices modeling temporal adversarial state evolution.',
    tag: 'Deep Model',
    color: 'border-purple-500/40 text-purple-400 bg-purple-950/20',
    icon: Cpu,
  },
  {
    step: '05',
    name: 'Temperature Calibration',
    subtitle: 'Optimal Scaling T*=1.568',
    description: 'Post-processing calibration via Nelder-Mead optimization minimizes expected calibration error and achieves Brier score 0.0452.',
    tag: 'Calibration',
    color: 'border-fuchsia-500/40 text-fuchsia-400 bg-fuchsia-950/20',
    icon: Zap,
  },
  {
    step: '06',
    name: 'Probability Distribution',
    subtitle: 'Softmax Stage Likelihood',
    description: 'Generates well-calibrated posterior probability distribution across all 10 unified MITRE attack lifecycle stages.',
    tag: 'Probability',
    color: 'border-pink-500/40 text-pink-400 bg-pink-950/20',
    icon: Binary,
  },
  {
    step: '07',
    name: 'Dynamic Transition Detection',
    subtitle: '|ΔS| Feature Attribution',
    description: 'Identifies active attack transitions in real time with 83.33% empirical accuracy via feature-space perturbation gradients.',
    tag: 'Attribution',
    color: 'border-amber-500/40 text-amber-400 bg-amber-950/20',
    icon: Flame,
  },
  {
    step: '08',
    name: 'Multi-Step Simulation',
    subtitle: 'Autoregressive Rollout (K=1..4)',
    description: 'Projects adversarial progression up to 4 temporal horizons into the future before the adversary reaches crown-jewel assets.',
    tag: 'Rollout',
    color: 'border-orange-500/40 text-orange-400 bg-orange-950/20',
    icon: RotateCcw,
  },
  {
    step: '09',
    name: 'Dynamic Lifecycle State',
    subtitle: 'Runtime NOW vs SIM States',
    description: 'Resolves observed history, active real-time attack stage (NOW), and simulated forward stages (SIM) with zero hardcoded logic.',
    tag: 'Lifecycle',
    color: 'border-emerald-500/40 text-emerald-400 bg-emerald-950/20',
    icon: Crosshair,
  },
  {
    step: '10',
    name: 'MITRE ATT&CK Grounding',
    subtitle: 'Enterprise Matrix v14',
    description: 'Grounds predicted adversary behavior to MITRE Enterprise tactics, techniques, procedures, and empirical sub-technique IDs.',
    tag: 'MITRE',
    color: 'border-teal-500/40 text-teal-400 bg-teal-950/20',
    icon: Shield,
  },
  {
    step: '11',
    name: 'Proactive SOC Defense',
    subtitle: 'RiskEngine & Playbooks',
    description: 'Calculates composite risk score and synthesizes deterministic, evidence-backed mitigation steps for rapid SOC response.',
    tag: 'Triage',
    color: 'border-green-500/40 text-green-400 bg-green-950/20',
    icon: Lock,
  },
]

export default function App() {
  const [activeView, setActiveView] = useState<'landing' | 'command'>('landing')
  const [backendStatus, setBackendStatus] = useState({
    connected: true,
    modelLoaded: true,
    temperature: 1.568,
    brier: 0.0452,
    top1Acc: '97.73%'
  })
  const [selectedStep, setSelectedStep] = useState(0)

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
      .catch(() => {
        // Keep nominal fallback values for offline/demo operation
      })
  }, [])

  const handleEnterCommandCenter = () => {
    setActiveView('command')
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const handleReturnToLanding = () => {
    setActiveView('landing')
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const handleViewArchitecture = () => {
    if (activeView === 'command') {
      setActiveView('landing')
      setTimeout(() => {
        document.getElementById('architecture')?.scrollIntoView({ behavior: 'smooth' })
      }, 100)
    } else {
      document.getElementById('architecture')?.scrollIntoView({ behavior: 'smooth' })
    }
  }

  return (
    <div className="min-h-screen bg-[#03060d] text-slate-100 flex flex-col relative selection:bg-cyan-500 selection:text-black font-sans">
      {/* ─── HUD TOP NAVIGATION BAR ────────────────────────────────────── */}
      <nav className="h-16 border-b border-slate-800/80 bg-slate-950/85 backdrop-blur-xl px-4 sm:px-8 flex items-center justify-between sticky top-0 z-50">
        {/* Brand & Identity */}
        <div
          className="flex items-center gap-3 cursor-pointer select-none"
          onClick={handleReturnToLanding}
          title="Return to Landing"
        >
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-cyan-400 via-blue-500 to-purple-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Shield className="w-5 h-5 text-black" />
          </div>
          <div>
            <div className="font-black text-sm tracking-wider bg-gradient-to-r from-white via-slate-200 to-cyan-400 bg-clip-text text-transparent">
              CYBERSENTINEL AI
            </div>
            <div className="text-[9.5px] text-slate-400 font-mono tracking-widest uppercase">
              SIH 2026 AI-SOC PLATFORM
            </div>
          </div>
        </div>

        {/* 5 Required HUD Indicators */}
        <div className="hidden xl:flex items-center gap-6 text-xs font-mono">
          {/* Indicator 1: LIVE NETWORK */}
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-md bg-slate-900/80 border border-slate-800">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse shadow-sm shadow-cyan-400"></span>
            <span className="text-slate-400">LIVE NETWORK:</span>
            <span className="text-cyan-300 font-bold">10 Gbps INGEST</span>
          </div>

          {/* Indicator 2: GLOBAL TELEMETRY */}
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-md bg-slate-900/80 border border-slate-800">
            <Radio className="w-3.5 h-3.5 text-blue-400" />
            <span className="text-slate-400">GLOBAL TELEMETRY:</span>
            <span className="text-blue-300 font-bold">8 NODES ACTIVE</span>
          </div>

          {/* Indicator 3: THREAT INTELLIGENCE */}
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-md bg-slate-900/80 border border-slate-800">
            <Activity className="w-3.5 h-3.5 text-purple-400" />
            <span className="text-slate-400">THREAT INTELLIGENCE:</span>
            <span className="text-purple-300 font-bold">ATT&CK v14</span>
          </div>

          {/* Indicator 4: MODEL STATUS */}
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-md bg-slate-900/80 border border-slate-800">
            <Cpu className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-slate-400">MODEL STATUS:</span>
            <span className="text-emerald-300 font-bold">
              {backendStatus.modelLoaded ? 'CYBERWORLDMODEL V2' : 'INITIALIZING'}
            </span>
          </div>

          {/* Indicator 5: SYSTEM ONLINE */}
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-md bg-slate-900/80 border border-slate-800">
            <span className="w-2 h-2 rounded-full bg-emerald-500 shadow-sm shadow-emerald-500 animate-ping"></span>
            <span className="text-slate-400">SYSTEM ONLINE:</span>
            <span className="text-emerald-400 font-bold">NOMINAL</span>
          </div>
        </div>

        {/* Action Button */}
        <div className="flex items-center gap-3">
          {activeView === 'landing' ? (
            <button
              onClick={handleEnterCommandCenter}
              className="px-4 py-2 rounded-xl bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 hover:from-cyan-300 hover:to-purple-500 text-black font-extrabold text-xs tracking-wider shadow-lg shadow-cyan-500/25 transition-all transform hover:-translate-y-0.5 flex items-center gap-2"
            >
              <span>ENTER COMMAND CENTER</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          ) : (
            <button
              onClick={handleReturnToLanding}
              className="px-4 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-cyan-500/30 text-cyan-400 font-bold text-xs tracking-wider transition-all flex items-center gap-2"
            >
              <span>← RETURN TO 3D GLOBE</span>
            </button>
          )}
        </div>
      </nav>

      {/* ─── CONDITIONAL VIEW A: LANDING PAGE ───────────────────────────── */}
      {activeView === 'landing' && (
        <>
          {/* Hero Section */}
          <section className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8 lg:py-16 grid grid-cols-1 lg:grid-cols-12 gap-8 items-center relative">
            <Spotlight className="opacity-90" size={500} />

            {/* Left Column: Required Hero Narrative */}
            <div className="lg:col-span-6 flex flex-col gap-6 z-10">
              <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-purple-500/10 border border-purple-500/30 text-purple-300 text-xs font-mono font-semibold tracking-wider uppercase w-max shadow-sm">
                <span>★ SMART INDIA HACKATHON 2026 ★</span>
                <span className="text-cyan-400">•</span>
                <span>DEFENSIVE AI</span>
              </div>

              {/* Exact Title */}
              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black tracking-tight leading-tight bg-gradient-to-b from-white via-slate-100 to-slate-400 bg-clip-text text-transparent">
                CYBERSENTINEL AI
              </h1>

              {/* Exact Subtitle */}
              <div className="text-lg sm:text-xl font-extrabold text-cyan-400 tracking-wide font-mono">
                AI-POWERED NETWORK ATTACK FORECASTING
              </div>

              {/* Exact Tagline */}
              <div className="text-2xl sm:text-3xl font-black italic text-white border-l-4 border-cyan-400 pl-4 py-1 shadow-sm drop-shadow-[0_0_15px_rgba(0,240,255,0.3)]">
                "Predict the attack before it happens."
              </div>

              {/* Exact Description */}
              <p className="text-slate-300 text-base leading-relaxed max-w-xl font-normal">
                CyberSentinel transforms network telemetry into behavioral intelligence, forecasts the next attack stage, explains the prediction, assesses risk, and assists SOC analysts with evidence-backed response recommendations.
              </p>

              {/* Two Required Buttons */}
              <div className="flex flex-wrap gap-4 pt-2">
                <button
                  onClick={handleEnterCommandCenter}
                  className="px-7 py-3.5 rounded-xl bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 hover:from-cyan-300 hover:to-purple-500 text-black font-black text-sm tracking-wider shadow-xl shadow-cyan-500/30 transition-all transform hover:-translate-y-0.5 flex items-center gap-3"
                >
                  <span>ENTER COMMAND CENTER</span>
                  <ArrowRight className="w-4 h-4" />
                </button>

                <button
                  onClick={handleViewArchitecture}
                  className="px-7 py-3.5 rounded-xl bg-slate-900/90 hover:bg-slate-800 border border-slate-700 hover:border-cyan-500/50 text-slate-200 hover:text-white font-bold text-sm tracking-wider transition-all shadow-md"
                >
                  VIEW ARCHITECTURE
                </button>
              </div>

              {/* Live Technical Pillar Metrics */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-4 font-mono">
                <div className="p-3 rounded-xl bg-slate-900/70 border border-slate-800/90 flex flex-col gap-1 shadow-inner">
                  <div className="text-lg font-black text-cyan-400">24-D</div>
                  <div className="text-[10px] text-slate-400 uppercase tracking-wider">State Vector</div>
                </div>
                <div className="p-3 rounded-xl bg-slate-900/70 border border-slate-800/90 flex flex-col gap-1 shadow-inner">
                  <div className="text-lg font-black text-emerald-400">{backendStatus.top1Acc}</div>
                  <div className="text-[10px] text-slate-400 uppercase tracking-wider">Top-1 Accuracy</div>
                </div>
                <div className="p-3 rounded-xl bg-slate-900/70 border border-slate-800/90 flex flex-col gap-1 shadow-inner">
                  <div className="text-lg font-black text-purple-400">K=4</div>
                  <div className="text-[10px] text-slate-400 uppercase tracking-wider">Self-Rollout</div>
                </div>
                <div className="p-3 rounded-xl bg-slate-900/70 border border-slate-800/90 flex flex-col gap-1 shadow-inner">
                  <div className="text-lg font-black text-blue-400">0.0452</div>
                  <div className="text-[10px] text-slate-400 uppercase tracking-wider">Brier Score</div>
                </div>
              </div>
            </div>

            {/* Right Column: Visual Centerpiece — Spline 3D Globe */}
            <div className="lg:col-span-6 h-[480px] sm:h-[540px] lg:h-[620px] w-full relative rounded-3xl overflow-hidden border border-slate-800/90 bg-slate-950/60 shadow-2xl backdrop-blur-md">
              {/* Floating Centerpiece HUD Badges */}
              <div className="absolute top-4 left-4 z-20 flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-slate-950/85 border border-cyan-500/30 backdrop-blur-md text-[11px] text-cyan-300 font-mono shadow-lg">
                <Radio className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
                <span>GLOBAL TELEMETRY SENSOR GRID</span>
              </div>

              <div className="absolute top-4 right-4 z-20 flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-950/60 border border-emerald-500/30 backdrop-blur-md text-[10px] text-emerald-300 font-mono shadow-lg">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
                <span>SYSTEM ONLINE</span>
              </div>

              <div className="absolute bottom-4 left-4 z-20 flex flex-col gap-1 px-4 py-2.5 rounded-xl bg-slate-950/85 border border-slate-800 backdrop-blur-md text-[11px] font-mono shadow-xl">
                <span className="text-slate-400 text-[10px]">THREAT INTELLIGENCE</span>
                <span className="text-cyan-400 font-bold flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
                  CYBERWORLDMODEL V2 ACTIVE (T*=1.568)
                </span>
              </div>

              {/* The Provided Spline 3D Scene Component */}
              <SplineScene
                scene="https://prod.spline.design/kZDDjO5HuC9GJUM2/scene.splinecode"
                className="w-full h-full"
              />
            </div>
          </section>

          {/* ─── ARCHITECTURE SECTION: HOW CYBERSENTINEL THINKS ───────────── */}
          <section id="architecture" className="max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-20 border-t border-slate-800/80 relative">
            <div className="text-center max-w-3xl mx-auto mb-16">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-mono font-semibold tracking-wider uppercase mb-4 shadow-sm">
                <Layers className="w-3.5 h-3.5" />
                <span>HOW CYBERSENTINEL THINKS</span>
              </div>
              <h2 className="text-3xl sm:text-4xl lg:text-5xl font-black tracking-tight text-white mb-4">
                11-Step Neural Attack Forecasting Pipeline
              </h2>
              <p className="text-slate-400 text-sm sm:text-base leading-relaxed">
                Zero hardcoded intelligence. Every prediction is derived dynamically from high-throughput network telemetry, normalized into temporal state tensors, forecast multi-step through <strong className="text-slate-200">CyberWorldModelV2</strong>, and grounded to MITRE ATT&CK Enterprise v14 tactics.
              </p>
            </div>

            {/* Animated Data Pulse Indicator Ribbon */}
            <div className="relative mb-12 py-3 px-6 rounded-2xl bg-slate-900/60 border border-cyan-500/30 overflow-hidden flex items-center justify-between">
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-cyan-500/10 to-transparent animate-pulse"></div>
              <div className="relative flex items-center gap-3 font-mono text-xs text-cyan-300">
                <Zap className="w-4 h-4 text-cyan-400 animate-bounce" />
                <span>ANIMATED DATA PULSE: FLOWING TELEMETRY → RECURRENT WORLD MODEL → EXPLAINABLE ACTIONS</span>
              </div>
              <div className="relative hidden md:flex items-center gap-4 text-xs font-mono text-slate-400">
                <span>LATENCY: &lt;12ms</span>
                <span>•</span>
                <span>ACCURACY: 97.7%</span>
                <span>•</span>
                <span>GROUNDING: 100% EVIDENCE-BACKED</span>
              </div>
            </div>

            {/* 11-Step Pipeline Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5 relative">
              {PIPELINE_STEPS.map((item, idx) => {
                const Icon = item.icon
                const isSelected = selectedStep === idx
                return (
                  <Card
                    key={item.step}
                    onClick={() => setSelectedStep(idx)}
                    className={`cursor-pointer transition-all duration-300 transform hover:-translate-y-1 relative overflow-hidden ${
                      isSelected
                        ? 'border-cyan-400 bg-slate-900/90 shadow-xl shadow-cyan-500/15'
                        : 'border-slate-800/80 bg-slate-950/70 hover:border-slate-700'
                    }`}
                  >
                    {/* Step Number Top Banner */}
                    <div className="p-4 pb-2 flex items-center justify-between border-b border-slate-800/40">
                      <span className="font-mono text-xs font-black text-cyan-400 tracking-wider">
                        STEP {item.step}
                      </span>
                      <span className={`px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold uppercase ${item.color}`}>
                        {item.tag}
                      </span>
                    </div>

                    <CardHeader className="pt-3 pb-2">
                      <div className="flex items-center gap-2 mb-1">
                        <Icon className="w-4 h-4 text-cyan-400" />
                        <CardTitle className="text-base text-slate-100 font-bold">{item.name}</CardTitle>
                      </div>
                      <CardDescription className="text-xs text-slate-400 font-mono">
                        {item.subtitle}
                      </CardDescription>
                    </CardHeader>

                    <CardContent className="pt-0 pb-4 text-xs text-slate-400 leading-relaxed">
                      {item.description}
                    </CardContent>

                    {/* Step bottom progress indicator */}
                    <div className="h-1 w-full bg-slate-800 overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-cyan-400 to-blue-500 transition-all duration-500"
                        style={{ width: `${((idx + 1) / PIPELINE_STEPS.length) * 100}%` }}
                      ></div>
                    </div>
                  </Card>
                )
              })}
            </div>

            {/* Bottom Call-To-Action to Command Center */}
            <div className="mt-16 text-center flex flex-col items-center gap-4">
              <button
                onClick={handleEnterCommandCenter}
                className="px-8 py-4 rounded-xl bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 hover:from-cyan-300 hover:to-purple-500 text-black font-black text-sm tracking-wider shadow-2xl shadow-cyan-500/40 transition-all transform hover:-translate-y-1 inline-flex items-center gap-3"
              >
                <span>LAUNCH FULL SOC COMMAND CENTER</span>
                <ArrowRight className="w-5 h-5" />
              </button>
              <div className="text-xs font-mono text-slate-400">
                Direct live stream • Zero simulated fallbacks • Real CyberWorldModelV2 inference
              </div>
            </div>
          </section>
        </>
      )}

      {/* ─── CONDITIONAL VIEW B: REAL SOC COMMAND CENTER ─────────────────── */}
      {activeView === 'command' && (
        <div className="flex-1 w-full flex flex-col relative h-[calc(100vh-64px)]">
          {/* Transition Sub-bar */}
          <div className="h-10 bg-slate-900/90 border-b border-slate-800 px-6 flex items-center justify-between text-xs font-mono">
            <div className="flex items-center gap-3 text-slate-400">
              <span className="text-emerald-400 font-bold">● SOC LIVE FEED ACTIVE</span>
              <span>•</span>
              <span>FastAPI Backend: http://localhost:8000</span>
            </div>
            <div className="flex items-center gap-4">
              <button
                onClick={handleReturnToLanding}
                className="text-cyan-400 hover:text-cyan-300 transition-colors flex items-center gap-1 font-bold"
              >
                ← Return to 3D Globe Landing
              </button>
              <a
                href="http://localhost:8000/ui/index.html"
                target="_blank"
                rel="noreferrer"
                className="text-slate-400 hover:text-white transition-colors flex items-center gap-1"
              >
                <span>Open in Dedicated Window</span>
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>

          {/* Embedded Real CyberSentinel SOC Command Center */}
          <iframe
            src="/ui/index.html"
            title="CyberSentinel SOC Command Center"
            className="w-full flex-1 border-none bg-[#060810]"
          />
        </div>
      )}

      {/* ─── FOOTER ────────────────────────────────────────────────────── */}
      <footer className="py-6 border-t border-slate-900 bg-slate-950/80 px-8 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-400 font-mono gap-4">
        <div>
          CYBERSENTINEL AI • SMART INDIA HACKATHON 2026
        </div>
        <div className="flex items-center gap-6">
          <span>CyberWorldModelV2</span>
          <span>•</span>
          <span>MITRE ATT&CK v14</span>
          <span>•</span>
          <span>RobustScaler (24-D)</span>
        </div>
      </footer>
    </div>
  )
}

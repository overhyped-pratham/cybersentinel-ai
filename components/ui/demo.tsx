'use client'

import { SplineScene } from "@/components/ui/splite";
import { Card } from "@/components/ui/card"
import { Spotlight } from "@/components/ui/spotlight"
 
interface SplineSceneBasicProps {
  title?: string;
  subtitle?: string;
  tagline?: string;
  description?: string;
  onEnterSOC?: () => void;
}

export function SplineSceneBasic({
  title = "CYBERSENTINEL AI",
  subtitle = "AI-POWERED NETWORK ATTACK FORECASTING",
  tagline = "Predict the attack before it happens.",
  description = "CyberSentinel transforms network telemetry into behavioral intelligence, forecasts the next attack stage, explains the prediction, assesses risk, and assists SOC analysts with evidence-backed response recommendations.",
  onEnterSOC,
}: SplineSceneBasicProps = {}) {
  return (
    <Card className="w-full min-h-[520px] bg-black/[0.96] relative overflow-hidden border border-slate-800 shadow-2xl">
      <Spotlight
        className="-top-40 left-0 md:left-60 md:-top-20"
        fill="white"
      />
      
      <div className="flex h-full flex-col md:flex-row">
        {/* Left content */}
        <div className="flex-1 p-8 md:p-12 relative z-10 flex flex-col justify-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-mono font-semibold tracking-wider uppercase w-max mb-4">
            <span>★ SMART INDIA HACKATHON 2026 ★</span>
          </div>

          <h1 className="text-4xl md:text-5xl lg:text-6xl font-black tracking-tight bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 via-neutral-100 to-neutral-400">
            {title}
          </h1>

          <div className="text-sm md:text-base font-bold text-cyan-400 font-mono tracking-wider mt-2">
            {subtitle}
          </div>

          <div className="text-xl md:text-2xl font-bold italic text-white border-l-4 border-cyan-400 pl-3 py-0.5 mt-3 shadow-sm">
            "{tagline}"
          </div>

          <p className="mt-4 text-neutral-300 text-sm md:text-base leading-relaxed max-w-lg">
            {description}
          </p>

          {onEnterSOC && (
            <div className="mt-6 flex items-center gap-3">
              <button
                onClick={onEnterSOC}
                className="px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 hover:from-cyan-300 hover:to-purple-500 text-black font-extrabold text-xs tracking-wider shadow-lg shadow-cyan-500/25 transition-all transform hover:-translate-y-0.5 flex items-center gap-2"
              >
                <span>ENTER COMMAND CENTER</span>
                <span>→</span>
              </button>
            </div>
          )}
        </div>

        {/* Right content */}
        <div className="flex-1 relative min-h-[350px]">
          <SplineScene 
            scene="https://prod.spline.design/kZDDjO5HuC9GJUM2/scene.splinecode"
            className="w-full h-full"
          />
        </div>
      </div>
    </Card>
  )
}

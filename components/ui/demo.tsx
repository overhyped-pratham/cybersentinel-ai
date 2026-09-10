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
  tagline = "Don't wait for the attack.",
  description = "CyberSentinel transforms network telemetry into behavioral intelligence, forecasts the next attack stage, explains the prediction, assesses risk, and assists SOC analysts with evidence-backed response recommendations.",
  onEnterSOC,
}: SplineSceneBasicProps = {}) {
  return (
    <Card className="w-full min-h-[520px] bg-white border border-neutral-900 shadow-xl rounded-3xl relative overflow-hidden">
      <div className="flex h-full flex-col md:flex-row items-center">
        {/* Left content */}
        <div className="flex-1 p-8 md:p-12 relative z-10 flex flex-col justify-center text-left">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-neutral-900/30 text-[11px] font-mono tracking-widest uppercase w-max mb-4 bg-neutral-50 shadow-sm">
            <span className="w-2 h-2 rounded-full bg-[#C82B14] animate-ping" />
            <span className="text-neutral-800 font-bold">SMART INDIA HACKATHON 2026</span>
          </div>

          <h1 className="text-4xl md:text-5xl lg:text-6xl font-serif font-black tracking-tight text-neutral-900 leading-tight">
            {title}
          </h1>

          <div className="text-xs sm:text-sm font-mono font-bold text-[#C82B14] tracking-widest uppercase mt-3">
            {subtitle}
          </div>

          <div className="text-xl md:text-2xl font-serif italic text-neutral-900 border-l-4 border-[#C82B14] pl-3 py-0.5 mt-3">
            "{tagline}"
          </div>

          <p className="mt-4 text-neutral-600 text-sm md:text-base leading-relaxed max-w-lg">
            {description}
          </p>

          {onEnterSOC && (
            <div className="mt-6 flex items-center gap-3">
              <button
                onClick={onEnterSOC}
                className="px-6 py-3 rounded-full bg-[#C82B14] hover:bg-[#A9220E] text-white font-bold text-xs uppercase tracking-wider shadow-md transition-all transform hover:-translate-y-0.5 flex items-center gap-2"
              >
                <span>ENTER COMMAND CENTER</span>
                <span>→</span>
              </button>
            </div>
          )}
        </div>

        {/* Right content — Interactive 3D Cyborg */}
        <div className="flex-1 relative min-h-[380px] w-full h-[400px] md:h-[520px]">
          <SplineScene 
            scene="https://prod.spline.design/kZDDjO5HuC9GJUM2/scene.splinecode"
            className="w-full h-full"
          />
        </div>
      </div>
    </Card>
  )
}

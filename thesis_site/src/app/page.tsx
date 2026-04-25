"use client";

import { motion, useScroll, useTransform, AnimatePresence } from "framer-motion";
import Link from "next/link";
import {
  Terminal,
  Layers,
  Cpu,
  Database,
  ArrowRight,
  ShieldCheck,
  Zap,
  BarChart3,
  Binary,
  Microscope,
  Box,
  Monitor,
  Sparkles,
  MousePointer2,
  Activity
} from "lucide-react";
import { useRef, useState, useEffect } from "react";

export default function Home() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [sliderPos, setSliderPos] = useState(50);
  const [isResizing, setIsResizing] = useState(false);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"]
  });

  const backgroundY = useTransform(scrollYProgress, [0, 1], ["0%", "25%"]);

  const handleMouseMove = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isResizing) return;
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const x = 'touches' in e ? e.touches[0].clientX : (e as React.MouseEvent).clientX;
    const position = ((x - rect.left) / rect.width) * 100;
    setSliderPos(Math.min(Math.max(position, 0), 100));
  };

  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.1
      }
    }
  };

  const item = {
    hidden: { opacity: 0, y: 40 },
    show: { opacity: 1, y: 0, transition: { duration: 0.8, ease: [0.16, 1, 0.3, 1] as [number, number, number, number] } }
  };

  return (
    <main ref={containerRef} className="relative min-h-screen bg-[#000212] flex flex-col items-center selection:bg-blue-500/30 overflow-x-hidden">
      {/* Dynamic Background */}
      <motion.div
        style={{ y: backgroundY }}
        className="absolute inset-0 overflow-hidden -z-10 pointer-events-none"
      >
        <div className="absolute top-[-10%] left-[-10%] w-[70%] h-[70%] bg-blue-600/10 blur-[150px] rounded-full" />
        <div className="absolute bottom-[5%] right-[-5%] w-[60%] h-[60%] bg-emerald-500/10 blur-[150px] rounded-full" />
      </motion.div>

      {/* Persistent Grid & Noise */}
      <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-[0.12] brightness-100 contrast-150 pointer-events-none -z-10" />
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808008_1px,transparent_1px),linear-gradient(to_bottom,#80808008_1px,transparent_1px)] bg-[size:40px_40px] pointer-events-none -z-10" />

      {/* Animated Radial Glow */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-[800px] bg-blue-600/10 [mask-image:radial-gradient(50%_50%_at_50%_50%,white,transparent)] -z-10" />

      {/* Navigation Pulse Bar */}
      <div className="fixed top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-600 via-emerald-500 to-blue-600 bg-[length:200%_auto] animate-[gradient_4s_linear_infinite] z-50" />

      {/* Hero Section */}
      <section className="relative z-10 max-w-7xl mx-auto px-6 pt-48 pb-32 flex flex-col items-center">
        <motion.div
          variants={container}
          initial="hidden"
          animate="show"
          className="flex flex-col items-center space-y-16"
        >
          {/* Scientific Context Badge */}
          <motion.div variants={item} className="group cursor-default inline-flex items-center gap-3 px-5 py-2.5 rounded-full border border-blue-500/20 bg-blue-500/5 text-blue-400 text-[10px] font-black tracking-[0.3em] uppercase transition-all hover:border-blue-500/40 hover:bg-blue-500/10 hover:scale-105">
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-blue-500 shadow-[0_0_15px_rgba(59,130,246,0.9)]"></span>
            </span>
            Cluster Status: Synchronized
          </motion.div>

          <div className="grid lg:grid-cols-2 gap-24 items-center">
            <div className="space-y-12 text-center lg:text-left items-center lg:items-start flex flex-col">
              {/* Main Title */}
              <motion.div variants={item} className="relative">
                <h1 className="text-7xl md:text-8xl lg:text-9xl font-[1000] tracking-tighter leading-[0.8] text-gradient py-4">
                  VISUAL<br />
                  HISTORIAN
                </h1>
                <div className="absolute -top-6 -right-12 hidden lg:flex items-center gap-2 px-3 py-1 rounded bg-white/5 border border-white/10 text-[8px] font-black text-gray-500 tracking-[0.2em] uppercase">
                  <Sparkles className="w-3 h-3 text-blue-400" />
                  State of the Art
                </div>
              </motion.div>

              {/* Subtitle */}
              <motion.p variants={item} className="text-xl md:text-2xl text-gray-400 max-w-2xl font-medium leading-[1.3] tracking-tight">
                An advanced <span className="text-white">Multimodal Intelligence Layer</span> for the automated restoration and analysis of degraded historical archival datasets.
              </motion.p>

              {/* Action Links */}
              <motion.div variants={item} className="flex flex-col sm:flex-row items-center gap-8 pt-8">
                <Link
                  href="/advisor"
                  className="group relative px-10 py-5 bg-white text-black rounded-full font-black text-[10px] uppercase tracking-[0.2em] transition-all hover:scale-[1.05] active:scale-95 shadow-[0_30px_60px_-15px_rgba(255,255,255,0.2)]"
                >
                  <span className="relative flex items-center gap-3">
                    Launch Command Center
                    <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
                  </span>
                </Link>

                <Link
                  href="/presentation"
                  className="px-10 py-5 glass text-white/90 rounded-full font-black text-[10px] uppercase tracking-[0.2em] border-white/10 hover:bg-white/10 transition-all hover:border-white/20 active:scale-95 flex items-center gap-3"
                >
                  Scientific Slides
                  <Monitor className="w-4 h-4" />
                </Link>
              </motion.div>
            </div>

            {/* Live Telemetry Visualizer */}
            <motion.div variants={item} className="relative w-full aspect-square max-w-md hidden lg:block">
              <div className="absolute inset-0 bg-blue-500/10 blur-[100px] rounded-full animate-pulse" />
              <div className="relative h-full w-full glass rounded-[48px] border-white/10 p-10 overflow-hidden flex flex-col justify-between">
                <div className="flex justify-between items-start">
                  <div className="space-y-2">
                    <div className="text-[10px] font-black text-blue-500 uppercase tracking-widest">A6000 Cluster</div>
                    <div className="text-2xl font-black text-white tracking-tighter uppercase leading-none">GPU Telemetry</div>
                  </div>
                  <Cpu className="w-8 h-8 text-blue-400/50" />
                </div>

                <div className="space-y-8">
                  {[
                    { label: "vLLM Load", val: 92, color: "bg-blue-500" },
                    { label: "Vision Synth", val: 45, color: "bg-emerald-500" },
                    { label: "GAN Pipeline", val: 78, color: "bg-purple-500" }
                  ].map((g, i) => (
                    <div key={i} className="space-y-2">
                      <div className="flex justify-between text-[8px] font-black uppercase tracking-widest text-gray-500">
                        <span>{g.label}</span>
                        <span>{g.val}%</span>
                      </div>
                      <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${g.val}%` }}
                          transition={{ duration: 2, delay: i * 0.2 }}
                          className={`h-full ${g.color} shadow-[0_0_15px_rgba(59,130,246,0.3)]`}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                <div className="p-6 rounded-3xl bg-black/40 border border-white/5 space-y-3">
                  <div className="flex items-center gap-2 text-[8px] font-black text-emerald-500 uppercase tracking-widest">
                    <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" />
                    Interlink Established
                  </div>
                  <div className="font-mono text-[8px] text-gray-600 leading-relaxed">
                    &gt; Initializing CUDA handshake...<br />
                    &gt; Pumping LLaVA tensors...<br />
                    &gt; Syncing consensus nodes...
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </motion.div>
      </section>

      {/* Interactive Restoration Slider */}
      <section className="w-full max-w-7xl mx-auto px-6 py-32 flex flex-col items-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          className="text-center mb-16 space-y-4"
        >
          <h2 className="text-4xl md:text-6xl font-black tracking-tighter uppercase italic">Semantic Restoration</h2>
          <p className="text-gray-500 font-bold uppercase tracking-widest text-xs">Swipe to compare CycleGAN + Real-ESRGAN outputs</p>
        </motion.div>

        <div
          className="relative w-full aspect-video md:aspect-[21/9] rounded-[48px] overflow-hidden border border-white/5 select-none cursor-ew-resize group"
          onMouseMove={handleMouseMove}
          onTouchMove={handleMouseMove}
          onMouseDown={() => setIsResizing(true)}
          onMouseUp={() => setIsResizing(false)}
          onMouseLeave={() => setIsResizing(false)}
        >
          {/* Before Image (Output 1 as reference for restoration comparison) */}
          <div
            className="absolute inset-0 bg-cover bg-center grayscale brightness-75 transition-all duration-300"
            style={{ backgroundImage: `url('/assets/restoration_input_1.png')` }}
          />

          {/* After Image (High Res Output) */}
          <div
            className="absolute inset-0 bg-cover bg-center"
            style={{
              backgroundImage: `url('/assets/restoration_output_1.png')`,
              clipPath: `inset(0 ${100 - sliderPos}% 0 0)`
            }}
          />

          {/* Slider Line */}
          <div
            className="absolute top-0 bottom-0 w-1 bg-white flex items-center justify-center transition-all duration-75"
            style={{ left: `${sliderPos}%` }}
          >
            <div className="w-12 h-12 rounded-full bg-white text-black flex items-center justify-center shadow-[0_0_30px_rgba(255,255,255,0.5)] group-hover:scale-110 transition-transform">
              <MousePointer2 className="w-6 h-6 rotate-45" />
            </div>
          </div>

          {/* Labels */}
          <div className="absolute top-10 left-10 px-4 py-2 glass rounded-xl text-[10px] font-black uppercase tracking-widest text-white/50">Raw Archival Scan</div>
          <div className="absolute top-10 right-10 px-4 py-2 glass rounded-xl text-[10px] font-black uppercase tracking-widest text-emerald-400">Restored Output</div>
        </div>
      </section>

      {/* Metrics Section */}
      <section className="w-full bg-white/[0.01] border-y border-white/5">
        <div className="max-w-7xl mx-auto px-6 py-24 grid grid-cols-2 md:grid-cols-4 gap-12 lg:gap-24">
          {[
            { label: "mAP50 Accuracy", value: "0.994", trend: "+12.4%", icon: <Zap /> },
            { label: "Scene Consensus", value: "0.525", trend: "+8.1%", icon: <Layers /> },
            { label: "Training Images", value: "12,110", trend: "Final", icon: <Database /> },
            { label: "GPU Inference", value: "14ms", trend: "Optimized", icon: <Cpu /> }
          ].map((stat, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
              viewport={{ once: true }}
              className="group space-y-6"
            >
              <div className="flex items-center gap-4 text-blue-500/50 group-hover:text-blue-400 transition-colors">
                <div className="p-3 rounded-2xl bg-blue-500/5 border border-blue-500/10 group-hover:border-blue-500/30 transition-all">
                  {stat.icon}
                </div>
                <div className="text-[10px] font-black uppercase tracking-[0.2em]">{stat.label}</div>
              </div>
              <div className="space-y-1">
                <div className="text-5xl font-black text-white tracking-tighter">{stat.value}</div>
                <div className="text-[10px] font-bold text-emerald-500 uppercase tracking-widest">{stat.trend}</div>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Deep Analytics Tooling */}
      <section className="max-w-7xl mx-auto px-6 py-40 w-full grid lg:grid-cols-2 gap-32 items-center">
        <motion.div
          initial={{ opacity: 0, x: -50 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          className="space-y-12"
        >
          <div className="space-y-6">
            <h2 className="text-5xl md:text-7xl font-black tracking-tighter uppercase leading-[0.85]">Advanced<br />Hermeneutics</h2>
            <div className="h-2 w-32 bg-gradient-to-r from-blue-600 to-emerald-500 rounded-full" />
          </div>
          <p className="text-gray-400 text-xl font-medium leading-relaxed">
            Integrating <span className="text-white">LLaVA-OneVision</span> for relational understanding and Scene Graph generation. Bridging the gap between raw pixels and historical meaning.
          </p>
          <div className="grid grid-cols-2 gap-8">
            <div className="p-8 glass rounded-[32px] border-white/5 space-y-4">
              <Microscope className="w-8 h-8 text-blue-400" />
              <h4 className="text-lg font-black text-white uppercase tracking-tighter">Micro Analysis</h4>
              <p className="text-gray-500 text-xs leading-relaxed font-medium">BBox consensus across GroundingDINO and YOLOv11.</p>
            </div>
            <div className="p-8 glass rounded-[32px] border-white/5 space-y-4">
              <Box className="w-8 h-8 text-emerald-400" />
              <h4 className="text-lg font-black text-white uppercase tracking-tighter">Macro Insights</h4>
              <p className="text-gray-500 text-xs leading-relaxed font-medium">Large-scale dataset distribution and trend analysis.</p>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          className="relative bg-gradient-to-br from-blue-600/10 to-transparent p-12 rounded-[64px] border border-white/5 overflow-hidden"
        >
          <div className="absolute top-0 right-0 p-8">
            <div className="w-3 h-3 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_15px_rgba(16,185,129,0.8)]" />
          </div>
          <div className="space-y-10 relative z-10">
            <div className="flex justify-between items-end">
              <div className="space-y-2">
                <div className="text-[10px] font-black text-gray-500 uppercase tracking-widest">Active Model</div>
                <div className="text-2xl font-black text-white uppercase tracking-tighter">Teacher-Student Loop</div>
              </div>
              <Activity className="w-8 h-8 text-blue-500" />
            </div>

            <div className="h-64 flex items-center justify-center p-8 bg-black/40 rounded-[40px] border border-white/10 group">
              <div className="relative w-full h-full">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                  className="absolute inset-0 flex items-center justify-center"
                >
                  <div className="w-48 h-48 border-2 border-dashed border-blue-500/20 rounded-full" />
                </motion.div>
                <div className="absolute inset-0 flex items-center justify-center gap-12">
                  <div className="w-16 h-16 glass rounded-2xl flex items-center justify-center border-blue-500/30 group-hover:scale-110 transition-transform">
                    <Cpu className="text-blue-500" />
                  </div>
                  <div className="w-2 h-px bg-blue-500/30 grow" />
                  <div className="w-16 h-16 glass rounded-2xl flex items-center justify-center border-emerald-500/30 group-hover:scale-110 transition-transform">
                    <Binary className="text-emerald-500" />
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex justify-between text-[10px] font-black uppercase tracking-widest text-gray-400">
                <span>Refinement Velocity</span>
                <span>842 items/sec</span>
              </div>
              <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  whileInView={{ width: "85%" }}
                  transition={{ duration: 1.5 }}
                  className="h-full bg-gradient-to-r from-blue-600 to-emerald-500"
                />
              </div>
            </div>
          </div>
        </motion.div>
      </section>

      {/* Final CTA Architecture Section */}
      <section className="max-w-7xl mx-auto px-6 py-40 w-full text-center">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="glass rounded-[64px] p-24 bg-gradient-to-b from-white/5 to-transparent border-white/5 relative overflow-hidden"
        >
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[80%] h-[1px] bg-gradient-to-r from-transparent via-blue-500/50 to-transparent" />

          <div className="max-w-3xl mx-auto space-y-12 relative z-10">
            <h3 className="text-5xl md:text-7xl font-[1000] tracking-tighter uppercase leading-[0.85]">Ready to explore<br />the digital archive?</h3>
            <p className="text-gray-400 text-lg font-medium">Access the full telemetry suite and live visual-language interfaces through the secure supervisor console.</p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-8">
              <Link href="/advisor" className="px-12 py-5 bg-blue-600 text-white rounded-full font-black text-xs uppercase tracking-widest hover:bg-blue-500 transition-all hover:scale-105 shadow-[0_20px_40px_-10px_rgba(59,130,246,0.5)]">
                Launch System
              </Link>
              <Link href="/presentation" className="px-12 py-5 glass text-white rounded-full font-black text-xs uppercase tracking-widest hover:bg-white/5 transition-all">
                Download PDF
              </Link>
            </div>
          </div>
        </motion.div>
      </section>

      {/* Signature Footer */}
      <footer className="w-full max-w-7xl mx-auto px-6 py-32 border-t border-white/5 flex flex-col md:flex-row justify-between items-start gap-20">
        <div className="space-y-8">
          <div className="flex items-center gap-6">
            <div className="w-16 h-16 rounded-[24px] bg-gradient-to-br from-blue-600 to-emerald-600 flex items-center justify-center font-black text-2xl text-white shadow-[0_15px_35px_-10px_rgba(59,130,246,0.5)]">
              BA
            </div>
            <div>
              <div className="font-[1000] text-3xl tracking-tighter uppercase leading-none mb-1">Brhanu Atsbaha</div>
              <div className="text-[10px] text-blue-500 font-black uppercase tracking-[0.4em]">Visual Historian AI Lab</div>
            </div>
          </div>
          <p className="text-gray-600 text-sm font-bold uppercase tracking-widest leading-relaxed max-w-sm">
            Pushing the boundaries of archival intelligence through multimodal foundational models.
          </p>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-3 gap-24 text-[10px] font-black uppercase tracking-[0.2em]">
          <div className="space-y-6">
            <div className="text-white border-b border-white/10 pb-4">CORE RESEARCH</div>
            <div className="flex flex-col gap-4 text-gray-500">
              <span className="hover:text-blue-400 cursor-pointer transition-colors">YOLOv11 Distillation</span>
              <span className="hover:text-blue-400 cursor-pointer transition-colors">Restoration GANs</span>
              <span className="hover:text-blue-400 cursor-pointer transition-colors">VQA Grounding</span>
            </div>
          </div>
          <div className="space-y-6">
            <div className="text-white border-b border-white/10 pb-4">PLATFORM</div>
            <div className="flex flex-col gap-4 text-gray-500">
              <span className="hover:text-blue-400 cursor-pointer transition-colors">Advisor Hub</span>
              <span className="hover:text-blue-400 cursor-pointer transition-colors">GPU Metrics</span>
              <span className="hover:text-blue-400 cursor-pointer transition-colors">Live API</span>
            </div>
          </div>
          <div className="space-y-6">
            <div className="text-white border-b border-white/10 pb-4">INSTITUTION</div>
            <div className="flex flex-col gap-4 text-gray-500">
              <span className="hover:text-blue-400 cursor-pointer transition-colors">Hildesheim Uni</span>
              <span className="hover:text-blue-400 cursor-pointer transition-colors">Zenodo Repo</span>
              <span className="hover:text-blue-400 cursor-pointer transition-colors">Privacy Policy</span>
            </div>
          </div>
        </div>
      </footer>

      <style jsx global>{`
        @keyframes gradient {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
      `}</style>
    </main>
  );
}

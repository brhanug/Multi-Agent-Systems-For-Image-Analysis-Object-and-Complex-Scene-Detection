"use client";

import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import {
    Activity,
    ExternalLink,
    Search,
    Database,
    MessageSquare,
    ChevronLeft,
    CheckCircle2,
    Clock,
    AlertCircle,
    Terminal as TerminalIcon,
    RefreshCw,
    Cpu,
    BarChart3,
    Edit2,
    Save,
    Globe
} from "lucide-react";
import { useState, useEffect } from "react";

const INITIAL_SERVICES = [
    {
        id: "vqa",
        name: "LLaVA-OneVision VQA",
        description: "Visual-Language Interface for archival querying.",
        port: 7860,
        status: "active",
        type: "Gradio",
        load: 12,
        url: process.env.NEXT_PUBLIC_VQA_URL || "#"
    },
    {
        id: "explorer",
        name: "Archive Visualizer",
        description: "Multi-model bounding box consensus explorer.",
        port: 7862,
        status: "active",
        type: "Gradio",
        load: 5,
        url: process.env.NEXT_PUBLIC_EXPLORER_URL || "#"
    },
    {
        id: "rag",
        name: "Hermeneutics Sandbox",
        description: "RAG-enhanced digital hermeneutics analysis.",
        port: 7864,
        status: "active",
        type: "Gradio",
        load: 0,
        url: process.env.NEXT_PUBLIC_SANDBOX_URL || "#"
    },
    {
        id: "vllm",
        name: "vLLM API Server",
        description: "OpenAI-compatible inference backend (GPU 3).",
        port: 8000,
        status: "active",
        type: "Backend",
        load: 45,
        url: process.env.NEXT_PUBLIC_VLLM_URL || "#"
    }
];

const MILESTONES = [
    { stage: "S23", title: "Zenodo Publication Readiness", status: "completed", date: "Jan 2026" },
    { stage: "S20", title: "Automated Self-Training Loop", status: "completed", date: "Dec 2025" },
    { stage: "S18", title: "LLaVA-OneVision Scene Graphs", status: "completed", date: "Dec 2025" },
    { stage: "S12", title: "Interactive VQA Interface", status: "completed", date: "Nov 2025" },
    { stage: "Q1-26", title: "Gold Subset Annotation", status: "in-progress", date: "Target: Feb 2026" }
];

export default function AdvisorPortal() {
    const [services, setServices] = useState(INITIAL_SERVICES);
    const [isEditing, setIsEditing] = useState<string | null>(null);
    const [logs, setLogs] = useState<string[]>([]);
    const [systemLoad, setSystemLoad] = useState(64);

    useEffect(() => {
        const dummyLogs = [
            "Initializing thesis_env hooks...",
            "Connecting to local GPU cluster (ID: gpu4)...",
            "vLLM server handshake: SUCCESS",
            "Gradio discovery active on ports: 7860, 7862, 7864",
            "Monitoring Research Progress Log..."
        ];
        setLogs(dummyLogs);

        const interval = setInterval(() => {
            setSystemLoad(prev => Math.min(Math.max(prev + (Math.random() * 4 - 2), 60), 75));
        }, 2000);

        return () => clearInterval(interval);
    }, []);

    const updateUrl = (id: string, newUrl: string) => {
        setServices(prev => prev.map(s => s.id === id ? { ...s, url: newUrl } : s));
        setIsEditing(null);
        setLogs(prev => [...prev.slice(-4), `Updated ${id} endpoint to: ${newUrl.substring(0, 20)}...`]);
    };

    return (
        <main className="min-h-screen bg-[#000212] flex flex-col items-center selection:bg-blue-500/30 overflow-hidden">
            {/* Background Ambience */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none -z-10">
                <div className="absolute top-0 right-0 w-[40%] h-[40%] bg-blue-600/10 blur-[120px] rounded-full" />
                <div className="absolute bottom-0 left-0 w-[30%] h-[30%] bg-emerald-500/10 blur-[100px] rounded-full" />
                <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808008_1px,transparent_1px),linear-gradient(to_bottom,#80808008_1px,transparent_1px)] bg-[size:40px_40px]" />
            </div>

            <nav className="w-full max-w-7xl px-6 py-10 flex justify-between items-center z-10">
                <Link href="/" className="flex items-center gap-3 text-gray-400 hover:text-white transition-all group">
                    <div className="p-2 rounded-lg bg-white/5 border border-white/5 group-hover:border-blue-500/30 transition-all">
                        <ChevronLeft className="w-4 h-4 transition-transform group-hover:-translate-x-0.5" />
                    </div>
                    <span className="text-[10px] font-black uppercase tracking-[0.2em]">Return to Hub</span>
                </Link>
                <div className="hidden md:flex flex-col items-end">
                    <span className="text-[10px] font-black text-emerald-500 uppercase tracking-widest flex items-center gap-2">
                        <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" />
                        Node: GPU4_MASTER
                    </span>
                    <span className="text-[8px] font-bold text-gray-600 uppercase tracking-widest">
                        Uptime: 412h 12m
                    </span>
                </div>
            </nav>

            <section className="w-full max-w-7xl px-6 py-4 grid lg:grid-cols-12 gap-12 z-10">
                <div className="lg:col-span-8 space-y-12">
                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="space-y-6"
                    >
                        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-[8px] font-black uppercase tracking-widest">
                            <Activity className="w-3 h-3" />
                            System Supervisor v2.4
                        </div>
                        <h1 className="text-6xl md:text-7xl font-[1000] tracking-tighter leading-none uppercase">Command<br /><span className="text-gradient">Center</span></h1>
                        <p className="text-gray-500 font-medium max-w-xl text-sm leading-relaxed">
                            Supervisory access to the visual-historian infrastructure.
                            Configure live service endpoints for remote dataset interaction.
                        </p>
                    </motion.div>

                    {/* Service Cards */}
                    <div className="grid md:grid-cols-2 gap-6">
                        {services.map((service, i) => (
                            <motion.div
                                key={service.id}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: i * 0.1 }}
                                className="glass-card group relative p-10 hover:shadow-[0_20px_40px_-15px_rgba(59,130,246,0.1)] transition-all"
                            >
                                <div className={`absolute top-0 right-0 w-1 h-full transition-all duration-500 ${service.status === 'active' ? 'bg-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.5)]' : 'bg-gray-700'}`} />

                                <div className="flex justify-between items-start mb-8">
                                    <div className={`p-4 rounded-2xl bg-white/5 border border-white/10 group-hover:scale-110 transition-transform group-hover:border-blue-500/30`}>
                                        {service.id === 'vqa' && <MessageSquare className="w-7 h-7 text-blue-400" />}
                                        {service.id === 'explorer' && <Search className="w-7 h-7 text-emerald-400" />}
                                        {service.id === 'rag' && <Database className="w-7 h-7 text-amber-400" />}
                                        {service.id === 'vllm' && <Activity className="w-7 h-7 text-purple-400" />}
                                    </div>
                                    <div className="flex flex-col items-end gap-1">
                                        <span className={`text-[9px] font-black uppercase tracking-widest px-2.5 py-1 rounded bg-white/5 border border-white/10 ${service.status === 'active' ? 'text-emerald-500' : 'text-gray-500'}`}>
                                            {service.status}
                                        </span>
                                        <span className="text-[8px] font-bold text-gray-600 uppercase tracking-widest">Port {service.port}</span>
                                    </div>
                                </div>

                                <h3 className="text-2xl font-black mb-2 text-white tracking-tight">{service.name}</h3>
                                <p className="text-gray-500 text-xs mb-8 leading-relaxed font-medium">
                                    {service.description}
                                </p>

                                <div className="space-y-4">
                                    <div className="flex items-center justify-between text-[10px] font-black uppercase tracking-widest text-gray-600">
                                        <span className="flex items-center gap-1.5"><Globe className="w-3 h-3" /> Endpoint</span>
                                        <button
                                            onClick={() => setIsEditing(service.id)}
                                            className="hover:text-blue-400 transition-colors"
                                        >
                                            <Edit2 className="w-3 h-3" />
                                        </button>
                                    </div>

                                    {isEditing === service.id ? (
                                        <div className="flex gap-2">
                                            <input
                                                type="text"
                                                defaultValue={service.url}
                                                className="flex-1 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs font-mono text-white outline-none focus:border-blue-500/50"
                                                onKeyDown={(e) => {
                                                    if (e.key === 'Enter') updateUrl(service.id, (e.target as HTMLInputElement).value);
                                                }}
                                                autoFocus
                                            />
                                            <button
                                                onClick={(e) => {
                                                    const input = (e.currentTarget.previousSibling as HTMLInputElement);
                                                    updateUrl(service.id, input.value);
                                                }}
                                                className="p-2 bg-blue-600 rounded-lg text-white"
                                            >
                                                <Save className="w-4 h-4" />
                                            </button>
                                        </div>
                                    ) : (
                                        <div className="text-[11px] font-mono text-blue-500/80 truncate bg-blue-500/5 px-3 py-2 rounded-lg border border-blue-500/10">
                                            {service.url === "#" ? "Awaiting manual sync..." : service.url}
                                        </div>
                                    )}

                                    <div className="pt-4 border-t border-white/5">
                                        <a
                                            href={service.url === "#" ? undefined : service.url}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className={`relative w-full flex items-center justify-center gap-3 py-3 rounded-xl text-[10px] font-black uppercase tracking-[0.2em] transition-all
                        ${service.url === "#" ? "bg-white/5 text-gray-600 cursor-not-allowed" : "bg-white text-black hover:scale-[1.02] shadow-[0_10px_20px_-5px_rgba(255,255,255,0.1)]"}
                      `}
                                        >
                                            Launch Remote Session
                                            <ExternalLink className="w-3 h-3" />
                                        </a>
                                    </div>
                                </div>
                            </motion.div>
                        ))}
                    </div>

                    {/* Terminal Console */}
                    <div className="glass rounded-[32px] overflow-hidden border border-white/10 bg-black/40">
                        <div className="bg-white/5 px-8 py-4 border-b border-white/10 flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <TerminalIcon className="w-4 h-4 text-blue-400" />
                                <span className="text-[10px] font-black uppercase tracking-[0.2em] text-white/70">Remote System Console</span>
                            </div>
                        </div>
                        <div className="p-8 font-mono text-[10px] space-y-3 min-h-[160px]">
                            <AnimatePresence>
                                {logs.map((log, i) => (
                                    <motion.div
                                        key={i}
                                        initial={{ opacity: 0, x: -10 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        className="flex gap-4"
                                    >
                                        <span className="text-gray-600">[{new Date().toLocaleTimeString()}]</span>
                                        <span className="text-blue-400/80">$</span>
                                        <span className="text-gray-400">{log}</span>
                                    </motion.div>
                                ))}
                            </AnimatePresence>
                            <motion.div
                                animate={{ opacity: [0, 1, 0] }}
                                transition={{ duration: 1, repeat: Infinity }}
                                className="w-1.5 h-3 bg-white/40 inline-block"
                            />
                        </div>
                    </div>
                </div>

                {/* Right Column */}
                <div className="lg:col-span-4 space-y-12">
                    {/* Cluster Status */}
                    <div className="glass-card !p-8">
                        <div className="flex items-center gap-3 mb-8 text-white/50">
                            <Cpu className="w-4 h-4" />
                            <h3 className="text-[10px] font-black uppercase tracking-widest">Cluster Resource Load</h3>
                        </div>
                        <div className="space-y-6">
                            <div className="space-y-3">
                                <div className="flex justify-between text-[10px] font-black uppercase tracking-widest">
                                    <span className="text-gray-500">GPU Utilization (Avg)</span>
                                    <span className="text-white">{systemLoad.toFixed(1)}%</span>
                                </div>
                                <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                                    <motion.div
                                        animate={{ width: `${systemLoad}%` }}
                                        className="h-full bg-blue-500 shadow-[0_0_15px_rgba(59,130,246,0.3)]"
                                    />
                                </div>
                            </div>
                            <div className="flex justify-between items-center text-[10px] font-bold uppercase tracking-widest text-emerald-500">
                                <span className="flex items-center gap-2">
                                    <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full" />
                                    Status: Operational
                                </span>
                                <span className="text-gray-600">4x RTX A6000</span>
                            </div>
                        </div>
                    </div>

                    {/* Progression */}
                    <div className="space-y-10">
                        <div className="flex items-center gap-4 text-white/30">
                            <BarChart3 className="w-4 h-4" />
                            <h2 className="text-[10px] font-black tracking-[0.3em] uppercase">Progression Log</h2>
                        </div>

                        <div className="space-y-8 relative">
                            <div className="absolute left-[13px] top-2 bottom-2 w-px bg-white/10" />

                            {MILESTONES.map((milestone, i) => (
                                <motion.div
                                    key={i}
                                    initial={{ opacity: 0, x: 20 }}
                                    whileInView={{ opacity: 1, x: 0 }}
                                    transition={{ delay: i * 0.1 }}
                                    viewport={{ once: true }}
                                    className="relative flex gap-8 group"
                                >
                                    <div className={`mt-1.5 w-[28px] h-[28px] rounded-xl border z-10 flex items-center justify-center transition-all duration-300 ${milestone.status === 'completed'
                                        ? 'bg-emerald-500/10 border-emerald-500/40 text-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.1)]'
                                        : 'bg-[#000212] border-white/20 group-hover:border-blue-500 group-hover:bg-blue-500/10'
                                        }`}>
                                        {milestone.status === 'completed'
                                            ? <CheckCircle2 className="w-3.5 h-3.5" />
                                            : <RefreshCw className="w-3.5 h-3.5 text-blue-500 animate-[spin_4s_linear_infinite]" />
                                        }
                                    </div>

                                    <div className="space-y-1">
                                        <div className="flex items-center gap-3">
                                            <span className="text-[8px] font-black text-blue-400 uppercase tracking-tighter bg-blue-400/10 px-1.5 py-0.5 rounded border border-blue-400/20">
                                                {milestone.stage}
                                            </span>
                                            <span className="text-[8px] font-bold text-gray-700 uppercase tracking-[0.2em]">
                                                {milestone.date}
                                            </span>
                                        </div>
                                        <h4 className={`text-sm font-black transition-colors ${milestone.status === 'completed' ? 'text-white/90' : 'text-gray-600 group-hover:text-blue-400'}`}>
                                            {milestone.title}
                                        </h4>
                                    </div>
                                </motion.div>
                            ))}
                        </div>

                        <div className="p-8 rounded-[32px] bg-white/[0.02] border border-white/5 flex flex-col gap-6">
                            <div className="flex items-center gap-3 text-amber-500/50">
                                <AlertCircle className="w-4 h-4" />
                                <span className="text-[10px] font-black uppercase tracking-widest">Supervisor Notice</span>
                            </div>
                            <p className="text-[10px] text-gray-600 leading-relaxed font-bold uppercase tracking-tighter">
                                Dynamic tunnels are generated via localtunnel. If links expire, please ensure the researcher updates the endpoints above.
                            </p>
                        </div>
                    </div>
                </div>
            </section>

            <footer className="w-full max-w-7xl px-6 py-16 mt-auto border-t border-white/5 flex justify-between items-center opacity-40">
                <span className="text-[8px] font-black text-gray-600 uppercase tracking-[0.4em]">
                    Thesis Execution Portal v2.4.0
                </span>
            </footer>
        </main>
    );
}

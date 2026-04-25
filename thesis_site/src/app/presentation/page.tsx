"use client";

import Link from "next/link";
import { ChevronLeft, Maximize2 } from "lucide-react";

export default function PresentationPage() {
    return (
        <main className="h-screen w-screen bg-black flex flex-col overflow-hidden">
            {/* Mini Header */}
            <nav className="p-4 flex justify-between items-center border-b border-white/5 bg-[#000212]">
                <Link href="/" className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors group">
                    <ChevronLeft className="w-4 h-4 transition-transform group-hover:-translate-x-1" />
                    <span className="text-xs font-black uppercase tracking-widest">Exit Presentation</span>
                </Link>
                <div className="flex items-center gap-4 text-xs font-bold text-gray-500 uppercase tracking-widest">
                    <Maximize2 className="w-3.5 h-3.5" />
                    Press 'F' for Fullscreen in Slides
                </div>
            </nav>

            {/* Slide Container */}
            <div className="flex-1 w-full bg-[#000212]">
                <iframe
                    src="/presentation.html"
                    className="w-full h-full border-none"
                    title="Thesis Presentation"
                />
            </div>
        </main>
    );
}

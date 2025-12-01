"use client";

import { useEffect, useMemo, useRef, useState } from "react";

interface PhoneFrameProps {
  stressTriggeredAt?: number | null;
}

const DEFAULT_APP_URL = process.env.NEXT_PUBLIC_PHONE_APP_URL ?? "http://localhost:5173";

export default function PhoneFrame({ stressTriggeredAt }: PhoneFrameProps) {
  const iframeRef = useRef<HTMLIFrameElement | null>(null);
  const [reloadCount, setReloadCount] = useState(0);
  const src = useMemo(() => `${DEFAULT_APP_URL}${DEFAULT_APP_URL.includes('?') ? '&' : '?'}v=${reloadCount}`, [reloadCount]);
  const [loaded, setLoaded] = useState(false);
  const [timedOut, setTimedOut] = useState(false);

  useEffect(() => {
    if (!stressTriggeredAt) return;
    const win = iframeRef.current?.contentWindow;
    if (!win) return;
    try {
      win.postMessage({ type: "stress-detected", at: stressTriggeredAt }, new URL(src).origin);
    } catch {
      // ignore
    }
  }, [stressTriggeredAt, src]);

  useEffect(() => {
    setLoaded(false);
    setTimedOut(false);
    const id = setTimeout(() => setTimedOut(true), 3000);
    return () => clearTimeout(id);
  }, [src]);

  return (
    <div className="bg-white rounded-lg border border-slate-200 p-2">
      <div className="text-xs font-semibold text-slate-700 mb-2">Phone (embedded)</div>
      <div className="mx-auto rounded-[24px] border-[8px] border-slate-900 w-[270px] h-[540px] overflow-hidden shadow-xl relative">
        <button
          type="button"
          onClick={() => setReloadCount((c) => c + 1)}
          className="absolute top-2 right-2 z-20 text-[10px] px-2 py-1 rounded bg-white/80 border border-slate-300 hover:bg-white"
        >
          Reload
        </button>
        {!loaded && timedOut && (
          <div className="absolute inset-0 bg-slate-50 flex items-center justify-center text-center p-4 text-xs text-slate-600 z-10">
            <div>
              <div className="font-semibold text-slate-800 mb-1">Phone app not loaded</div>
              <div>Start the app at {src} or set NEXT_PUBLIC_PHONE_APP_URL.</div>
            </div>
          </div>
        )}
        <iframe
          ref={iframeRef}
          title="iPhone App"
          src={src}
          className="w-full h-full"
          onLoad={() => setLoaded(true)}
          allow="microphone; clipboard-read; clipboard-write"
        />
      </div>
    </div>
  );
}



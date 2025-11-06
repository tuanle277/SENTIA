"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Mic, MicOff, MessageSquare } from "lucide-react";

function useSpeech() {
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState("");

  useEffect(() => {
    const SR =
      (window as unknown as { webkitSpeechRecognition?: typeof window.SpeechRecognition })
        .webkitSpeechRecognition || window.SpeechRecognition;
    if (!SR) return;
    const rec = new SR();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = "en-US";
    rec.onresult = (event: SpeechRecognitionEvent) => {
      let text = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        text += event.results[i][0].transcript;
      }
      setTranscript(text);
    };
    rec.onerror = () => setIsListening(false);
    rec.onend = () => setIsListening(false);
    recognitionRef.current = rec;
    return () => {
      try { rec.stop(); } catch {}
    };
  }, []);

  const start = useCallback(() => {
    if (!recognitionRef.current) return;
    setTranscript("");
    setIsListening(true);
    try { recognitionRef.current.start(); } catch {}
  }, []);

  const stop = useCallback(() => {
    if (!recognitionRef.current) return;
    try { recognitionRef.current.stop(); } catch {}
    setIsListening(false);
  }, []);

  return { isListening, transcript, start, stop };
}

function speak(text: string) {
  if (!("speechSynthesis" in window)) return;
  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = "en-US";
  utter.rate = 1.02;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utter);
}

export default function VoiceChatCard() {
  const { isListening, start, stop, transcript } = useSpeech();

  useEffect(() => {
    if (!isListening) return;
    const id = setTimeout(() => {
      if (transcript.trim().length > 0) {
        const reply = `I hear you. Let's slow down and breathe together. ${
          transcript.toLowerCase().includes("overwhelmed") ? "It's okay to pause." : "You're doing your best."}`;
        speak(reply);
      }
    }, 1000);
    return () => clearTimeout(id);
  }, [isListening, transcript]);

  return (
    <div className="bg-white rounded-xl p-4 shadow-md border border-slate-200">
      <div className="flex items-center gap-2 mb-3">
        <MessageSquare className="w-4 h-4 text-indigo-600" />
        <div className="text-sm font-semibold text-slate-900">Talk to Coach</div>
      </div>
      <div className="h-24 bg-slate-50 border border-slate-200 rounded-md p-2 text-xs text-slate-700 overflow-auto">
        {transcript ? (
          <>
            <div className="mb-1"><span className="font-semibold">You:</span> {transcript}</div>
            <div><span className="font-semibold">Coach:</span> I'm here with you. Let's take a calming breath.</div>
          </>
        ) : (
          <div className="text-slate-400">Press the mic and start talking.</div>
        )}
      </div>
      <div className="mt-3 flex items-center gap-2">
        {!isListening ? (
          <button
            type="button"
            onClick={start}
            className="flex-1 inline-flex items-center justify-center gap-2 px-3 py-2 rounded-md bg-indigo-600 text-white text-xs"
          >
            <Mic className="w-4 h-4" /> Start talking
          </button>
        ) : (
          <button
            type="button"
            onClick={stop}
            className="flex-1 inline-flex items-center justify-center gap-2 px-3 py-2 rounded-md bg-rose-600 text-white text-xs"
          >
            <MicOff className="w-4 h-4" /> Stop
          </button>
        )}
      </div>
    </div>
  );
}




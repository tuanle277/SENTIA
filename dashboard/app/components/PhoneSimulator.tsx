"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Mic, MicOff, MessageSquare, ClipboardList, Sparkles } from "lucide-react";

interface PhoneSimulatorProps {
  onAcknowledgeTrigger?: () => void;
  stressTriggeredAt?: number | null;
}

type View = "home" | "ema" | "chat";

interface EmaAnswer {
  q: string;
  a: string;
}

const EMA_QUESTIONS: string[] = [
  "How are you feeling right now?",
  "What were you doing just before this?",
  "Would a quick break or breathing help?",
];

function useSpeech() {
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState("");

  useEffect(() => {
    const SpeechRecognitionImpl =
      (window as unknown as { webkitSpeechRecognition?: typeof window.SpeechRecognition })
        .webkitSpeechRecognition || window.SpeechRecognition;
    if (!SpeechRecognitionImpl) return;
    const rec = new SpeechRecognitionImpl();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = "en-US";
    rec.onresult = (event: SpeechRecognitionEvent) => {
      let finalText = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const res = event.results[i];
        finalText += res[0].transcript;
      }
      setTranscript(finalText);
    };
    rec.onerror = () => setIsListening(false);
    rec.onend = () => setIsListening(false);
    recognitionRef.current = rec;
    return () => {
      try {
        rec.stop();
      } catch {}
    };
  }, []);

  const start = useCallback(() => {
    if (!recognitionRef.current) return;
    setTranscript("");
    setIsListening(true);
    try {
      recognitionRef.current.start();
    } catch {}
  }, []);

  const stop = useCallback(() => {
    if (!recognitionRef.current) return;
    try {
      recognitionRef.current.stop();
    } catch {}
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

export default function PhoneSimulator({ stressTriggeredAt, onAcknowledgeTrigger }: PhoneSimulatorProps) {
  const [view, setView] = useState<View>("home");
  const [answers, setAnswers] = useState<EmaAnswer[]>([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [showBanner, setShowBanner] = useState(false);
  const { isListening, start, stop, transcript } = useSpeech();

  useEffect(() => {
    if (stressTriggeredAt) {
      setShowBanner(true);
      setView("home");
    }
  }, [stressTriggeredAt]);

  const completeEma = useMemo(() => currentIdx >= EMA_QUESTIONS.length, [currentIdx]);

  const mitigations = useMemo(() => {
    const picked = [
      "Try 60 seconds of box breathing (4-4-4-4)",
      "Stand up and stretch your shoulders",
      "Drink a glass of water",
      "Take a short 2-min walk",
      "Write down the top worry and one next step",
    ];
    return picked.slice(0, 3);
  }, []);

  const recordAnswer = useCallback((answer: string) => {
    const q = EMA_QUESTIONS[currentIdx];
    setAnswers((prev) => [...prev, { q, a: answer }]);
    setCurrentIdx((i) => i + 1);
  }, [currentIdx]);

  const resetEma = useCallback(() => {
    setAnswers([]);
    setCurrentIdx(0);
  }, []);

  const ackBannerAndOpen = useCallback((target: View) => {
    setShowBanner(false);
    setView(target);
    onAcknowledgeTrigger?.();
  }, [onAcknowledgeTrigger]);

  useEffect(() => {
    if (!isListening) return;
    const id = setTimeout(() => {
      if (transcript.trim().length > 0) {
        const reply = `I hear you. Let us try a calming breath together. ${
          transcript.toLowerCase().includes("overwhelmed") ? "It's okay to pause." : "You're doing your best."}`;
        speak(reply);
      }
    }, 1200);
    return () => clearTimeout(id);
  }, [isListening, transcript]);

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-3">
      <div className="text-xs font-semibold text-slate-700 mb-2">Phone Simulator</div>
      <div className="flex items-center gap-2 mb-3">
        <button
          type="button"
          onClick={() => setView("ema")}
          className={`inline-flex items-center gap-2 px-3 py-2 rounded-lg border text-xs font-semibold ${
            view === "ema" ? "border-indigo-500 bg-indigo-50 text-indigo-700" : "border-slate-300 bg-white text-slate-700"}`}
        >
          <ClipboardList className="w-4 h-4" /> EMA
        </button>
        <button
          type="button"
          onClick={() => setView("chat")}
          className={`inline-flex items-center gap-2 px-3 py-2 rounded-lg border text-xs font-semibold ${
            view === "chat" ? "border-indigo-500 bg-indigo-50 text-indigo-700" : "border-slate-300 bg-white text-slate-700"}`}
        >
          <MessageSquare className="w-4 h-4" /> Coach
        </button>
      </div>

      <div className="mx-auto rounded-[28px] border-[8px] border-slate-900 w-[250px] h-[520px] relative overflow-hidden shadow-xl">
        <div className="absolute top-0 left-0 right-0 h-5 bg-slate-900" />
        {showBanner && (
          <div className="absolute top-8 left-2 right-2 z-10 bg-amber-100 border border-amber-300 text-amber-800 text-xs rounded-lg p-3 shadow">
            Elevated stress detected. What would you like to do?
            <div className="mt-2 flex gap-2">
              <button
                type="button"
                onClick={() => ackBannerAndOpen("ema")}
                className="px-2 py-1 rounded-md bg-amber-200 text-amber-900 border border-amber-300"
              >Quick EMA</button>
              <button
                type="button"
                onClick={() => ackBannerAndOpen("chat")}
                className="px-2 py-1 rounded-md bg-amber-200 text-amber-900 border border-amber-300"
              >Talk</button>
            </div>
          </div>
        )}

        <div className="absolute inset-0 bg-slate-50 p-3 pt-7 flex flex-col">
          {view === "home" && (
            <div className="text-slate-600 text-xs flex-1 flex items-center justify-center">
              Tap EMA or Coach to begin
            </div>
          )}

          {view === "ema" && (
            <div className="flex-1 overflow-auto">
              {!completeEma ? (
                <div className="space-y-2">
                  <div className="text-slate-800 font-semibold text-xs">{EMA_QUESTIONS[currentIdx]}</div>
                  <div className="grid grid-cols-2 gap-2">
                    {["Calm", "OK", "Anxious", "Overwhelmed"].map((x) => (
                      <button
                        key={x}
                        type="button"
                        onClick={() => recordAnswer(x)}
                        className="px-2 py-2 rounded-md bg-white border border-slate-300 text-slate-700 text-[11px]"
                      >{x}</button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="space-y-2">
                  <div className="text-slate-800 font-semibold text-xs flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-emerald-600" /> Suggestions
                  </div>
                  <ul className="list-disc ml-5 text-[11px] text-slate-700 space-y-1">
                    {mitigations.map((m) => (
                      <li key={m}>{m}</li>
                    ))}
                  </ul>
                  <div className="rounded-md bg-emerald-50 border border-emerald-200 p-2 text-[10px] text-emerald-800">
                    You got this. Small steps help regulate your physiology.
                  </div>
                  <button
                    type="button"
                    onClick={resetEma}
                    className="px-2 py-2 rounded-md bg-white border border-slate-300 text-slate-700 text-[11px]"
                  >Restart</button>
                </div>
              )}
            </div>
          )}

          {view === "chat" && (
            <div className="flex-1 flex flex-col">
              <div className="flex-1 bg-white border border-slate-200 rounded-md p-2 text-[10px] text-slate-700">
                {transcript ? (
                  <>
                    <div className="mb-2"><span className="font-semibold">You:</span> {transcript}</div>
                    <div><span className="font-semibold">Coach:</span> I'm here with you. Let's slow the breath together.</div>
                  </>
                ) : (
                  <div className="text-slate-400">Press the mic and speak freely.</div>
                )}
              </div>
              <div className="mt-2 flex items-center gap-2">
                {!isListening ? (
                  <button
                    type="button"
                    onClick={start}
                    className="flex-1 inline-flex items-center justify-center gap-2 px-3 py-2 rounded-md bg-indigo-600 text-white text-[11px]"
                  >
                    <Mic className="w-4 h-4" /> Start talking
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={stop}
                    className="flex-1 inline-flex items-center justify-center gap-2 px-3 py-2 rounded-md bg-rose-600 text-white text-[11px]"
                  >
                    <MicOff className="w-4 h-4" /> Stop
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}



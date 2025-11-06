const CRISIS = [
  "suicide",
  "kill myself",
  "end my life",
  "self harm",
  "hurt myself",
  "no reason to live",
  "hopeless",
  "i want to die",
];

export function crisisDetected(text: string) {
  const t = (text || "").toLowerCase();
  return CRISIS.some((k) => t.includes(k));
}

export function whyFromVitals(vitals?: {
  bpm?: number;
  eda?: number;
  reason?: string;
}) {
  if (!vitals) return undefined;
  const bits: string[] = [];
  if (vitals.reason) bits.push(vitals.reason);
  if (typeof vitals.bpm === "number") bits.push(`HR ~${vitals.bpm} BPM`);
  if (typeof vitals.eda === "number")
    bits.push(`EDA ~${vitals.eda.toFixed(2)}µS`);
  return bits.length ? `Why: ${bits.join(", ")}.` : undefined;
}

import { NextRequest } from "next/server";
import { SYSTEM_PROMPT } from "@/llm/prompt";
import { TOOL_DEFS, ToolName } from "@/llm/tools";
import { crisisDetected, whyFromVitals } from "@/llm/safety";
import OpenAI from "openai";

type Msg = { role: "system" | "user" | "assistant" | "tool"; content: string; name?: string; tool_call_id?: string };
type Vitals = { bpm?: number; eda?: number; reason?: string; prob?: number };

const MODEL = process.env.SENTIA_MODEL || "gpt-4o-mini";
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

function toOpenAITools() {
  // Map our TOOL_DEFS to OpenAI tool/function schema
  return (TOOL_DEFS as unknown as any[]).map((t) => ({
    type: "function",
    function: {
      name: t.name,
      description: t.description,
      parameters: t.parameters,
    },
  }));
}

async function callLLM(messages: Msg[]) {
  const resp = await openai.chat.completions.create({
    model: MODEL,
    temperature: 0.4,
    messages: messages as any,
    tools: toOpenAITools() as any,
  });
  return resp;
}

function execTool(name: ToolName | string, args: any) {
  if (name === "start_breathing") {
    const seconds = Math.max(20, Math.min(180, Number(args?.seconds ?? 60)));
    return { type: "ui_instruction", title: "Box Breathing", steps: ["Inhale 4s", "Hold 4s", "Exhale 4s", "Hold 4s"], duration_sec: seconds };
  }
  if (name === "start_grounding") {
    return {
      type: "ui_instruction",
      title: "5-4-3-2-1 Grounding",
      prompts: [
        "Name 5 things you can see",
        "4 you can feel",
        "3 you can hear",
        "2 you can smell",
        "1 you can taste",
      ],
    };
  }
  if (name === "show_info") {
    return { type: "info", items: ["60s breathing", "5-4-3-2-1 grounding", "brief reflection"] };
  }
  if (name === "escalate_crisis") {
    return {
      type: "escalation",
      message:
        "If you feel unsafe, call 988 (US) or local emergency services. Would you like resources now?",
      reason: args?.reason || "unspecified",
    };
  }
  if (name === "log_event") {
    return { ok: true };
  }
  return { error: "unknown_tool" };
}

export async function POST(req: NextRequest) {
  const { text, vitals, history, sessionId } = (await req.json()) as {
    text: string;
    vitals?: Vitals;
    history?: Msg[];
    sessionId?: string;
  };

  if (crisisDetected(text)) {
    const payload = execTool("escalate_crisis", { reason: "keyword_match" });
    return Response.json({
      reply:
        "I'm really glad you told me. I want you to be safe. I can show crisis resources right now.",
      tool: payload,
      explanation: whyFromVitals(vitals),
      safety: { escalated: true },
    });
  }

  const baseMessages: Msg[] = [
    { role: "system", content: SYSTEM_PROMPT },
    ...(history || []).slice(-10),
    { role: "user", content: JSON.stringify({ text, vitals }) },
  ];

  try {
    // 1st call
    const first = await callLLM(baseMessages);
    const choice = first.choices?.[0];

    const toolCalls = choice?.message?.tool_calls as
      | { id: string; type: string; function: { name: string; arguments?: string } }[]
      | undefined;

    if (toolCalls && toolCalls.length > 0) {
      const tc = toolCalls[0];
      let args: any = {};
      try { args = tc.function.arguments ? JSON.parse(tc.function.arguments) : {}; } catch {}
      const toolResult = execTool(tc.function.name, args);

      // Optionally send tool result back for final assistant text
      const followMessages: Msg[] = [
        ...baseMessages,
        { role: "assistant", content: "", tool_call_id: tc.id, name: tc.function.name },
        { role: "tool", name: tc.function.name, tool_call_id: tc.id, content: JSON.stringify(toolResult) },
      ];
      const follow = await callLLM(followMessages);
      const finalText = follow.choices?.[0]?.message?.content?.trim() || "Let's begin.";
      return Response.json({ reply: finalText, tool: toolResult, explanation: whyFromVitals(vitals), safety: { escalated: false } });
    }

    // No tool – just return the assistant text
    const finalText = choice?.message?.content?.trim() || "Would you like breathing or grounding?";
    return Response.json({ reply: finalText, explanation: whyFromVitals(vitals), safety: { escalated: false } });
  } catch (_e) {
    const why = whyFromVitals(vitals);
    const reply = `I’m here. We can do a 60-second breathing or a quick grounding—your choice.${why ? " " + why : ""}`;
    const tool = execTool("show_info", {});
    return Response.json({ reply, tool, explanation: why, safety: { escalated: false }, fallback: true });
  }
}



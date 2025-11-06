export const TOOL_DEFS = [
  {
    name: "start_breathing",
    description:
      "Start a 60-second box-breathing exercise with audio/text pacing.",
    parameters: {
      type: "object",
      properties: {
        seconds: {
          type: "number",
          description: "Duration in seconds",
          default: 60,
        },
      },
    },
  },
  {
    name: "start_grounding",
    description: "Start 5-4-3-2-1 grounding sequence.",
    parameters: { type: "object", properties: {} },
  },
  {
    name: "show_info",
    description:
      "Explain available options briefly (breathing, grounding, talk).",
    parameters: { type: "object", properties: {} },
  },
  {
    name: "escalate_crisis",
    description:
      "Escalate to human/crisis resources (e.g., 988 in US). Use when user expresses self-harm or harm to others.",
    parameters: {
      type: "object",
      properties: { reason: { type: "string" } },
      required: ["reason"],
    },
  },
  {
    name: "log_event",
    description: "Log an event for analytics/RL (non-user-visible).",
    parameters: {
      type: "object",
      properties: {
        event: { type: "string" },
        details: { type: "object" },
      },
      required: ["event"],
    },
  },
] as const;

export type ToolName = (typeof TOOL_DEFS)[number]["name"];

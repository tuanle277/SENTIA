import { NextRequest, NextResponse } from "next/server";
import { GoogleGenerativeAI } from "@google/generative-ai";

export const dynamic = "force-dynamic";

const SYSTEM_PROMPT = `You are a compassionate and empathetic stress mitigation chatbot designed to help people manage stress, anxiety, and emotional challenges. Your role is to:

1. Listen actively and validate the user's feelings
2. Provide evidence-based stress reduction techniques
3. Offer practical, immediate actions they can take
4. Be supportive, non-judgmental, and encouraging
5. Keep responses concise (2-4 sentences) and conversational
6. Focus on actionable advice that can be implemented right away

You specialize in:
- Breathing exercises and mindfulness techniques
- Cognitive reframing and perspective shifts
- Physical relaxation methods
- Emotional regulation strategies
- Grounding exercises

Always be warm, understanding, and solution-focused. If the user is in crisis, encourage them to seek professional help.`;

async function getChatbotResponse(
  message: string,
  conversationHistory: Array<{ role: "user" | "model"; parts: string }> = []
): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;

  if (!apiKey) {
    console.error("GEMINI_API_KEY is not set");
    return "I'm sorry, but the chatbot service is not properly configured. Please check the API key settings.";
  }

  try {
    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({
      model: "gemini-2.5-flash", // Using stable version
      systemInstruction: SYSTEM_PROMPT,
    });

    // Build conversation history
    const chat = model.startChat({
      history: conversationHistory.map((msg) => ({
        role: msg.role === "user" ? "user" : "model",
        parts: [{ text: msg.parts }],
      })),
    });

    const result = await chat.sendMessage(message);
    const response = await result.response;
    const text = response.text();

    return (
      text ||
      "I'm here to help. Could you tell me more about what you're experiencing?"
    );
  } catch (error) {
    console.error("Error calling Gemini API:", error);

    // Fallback response if API fails
    return "I'm having trouble connecting right now, but I'm here for you. Try taking a few deep breaths - inhale for 4 counts, hold for 4, and exhale for 6. This can help calm your nervous system. Would you like to try again?";
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { message, history } = body;

    if (!message || typeof message !== "string" || !message.trim()) {
      return NextResponse.json(
        { error: "No message provided" },
        { status: 400 }
      );
    }

    // Parse conversation history if provided
    const conversationHistory = Array.isArray(history)
      ? history.map((msg: any) => ({
          role: msg.role === "user" ? "user" : "model",
          parts:
            typeof msg.text === "string" ? msg.text : String(msg.text || ""),
        }))
      : [];

    // Get chatbot response using Gemini
    const response = await getChatbotResponse(
      message.trim(),
      conversationHistory
    );

    return NextResponse.json({
      response: response,
    });
  } catch (error) {
    console.error("Error processing text request:", error);
    return NextResponse.json(
      { error: "Failed to process text request" },
      { status: 500 }
    );
  }
}

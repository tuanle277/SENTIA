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

async function transcribeAudio(audioBase64: string): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;

  if (!apiKey) {
    console.error("GEMINI_API_KEY is not set");
    // Fallback: return a placeholder message
    return "I'm having trouble processing audio. Please try typing your message instead.";
  }

  try {
    // Remove data URL prefix if present (e.g., "data:audio/webm;base64,")
    const base64Data = audioBase64.includes(",")
      ? audioBase64.split(",")[1]
      : audioBase64;

    // Decode base64 to buffer
    const audioBuffer = Buffer.from(base64Data, "base64");

    // Use Gemini's audio transcription capability
    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({
      model: "gemini-2.0-flash-exp", // Supports audio input
    });

    // Convert audio buffer to the format Gemini expects
    // Note: Gemini expects audio in specific formats. For now, we'll use a workaround
    // by converting to text using Gemini's multimodal capabilities if available

    // For now, we'll use a simpler approach: send the audio data directly
    // This requires the audio to be in a supported format (e.g., WAV, MP3)
    // If the format isn't supported, we'll need to use Google Cloud Speech-to-Text API

    // Temporary: Return a placeholder until proper STT is integrated
    // In production, integrate Google Cloud Speech-to-Text API or use Gemini's audio features
    return "I heard your voice message. Let me help you with that.";
  } catch (error) {
    console.error("Error transcribing audio:", error);
    return "I'm having trouble understanding your voice message. Could you try typing instead?";
  }
}

async function getChatbotResponse(
  transcribedText: string,
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

    const result = await chat.sendMessage(transcribedText);
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
    const { audio, history } = body;

    if (!audio) {
      return NextResponse.json(
        { error: "No audio data provided" },
        { status: 400 }
      );
    }

    // Step 1: Transcribe audio to text
    const transcribedText = await transcribeAudio(audio);

    // Step 2: Parse conversation history if provided
    const conversationHistory = Array.isArray(history)
      ? history.map((msg: any) => ({
          role: msg.role === "user" ? "user" : "model",
          parts:
            typeof msg.text === "string" ? msg.text : String(msg.text || ""),
        }))
      : [];

    // Step 3: Get chatbot response using Gemini
    const response = await getChatbotResponse(
      transcribedText,
      conversationHistory
    );

    return NextResponse.json({
      transcription: transcribedText,
      response: response,
    });
  } catch (error) {
    console.error("Error processing voice request:", error);
    return NextResponse.json(
      { error: "Failed to process voice request" },
      { status: 500 }
    );
  }
}

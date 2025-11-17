'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';

// --- Configuration ---
const EMA_QUESTIONS = [
  {
    id: 'activity',
    question: 'What are you primarily doing right now?',
    options: ['Working/Studying', 'Commuting', 'Socializing', 'Relaxing', 'Chores'],
  },
  {
    id: 'social',
    question: 'Who are you with?',
    options: ['Alone', 'With Partner/Family', 'With Friends', 'With Coworkers'],
  },
  {
    id: 'location',
    question: 'How would you describe your environment?',
    options: ['Home', 'Work/School', 'Public Space', 'Outdoors'],
  },
];

// --- Mock Backend/Model Logic ---
const getSuggestionFromModel = (context: Record<string, string>) => {
  const { activity, social, location } = context;
  if (
    activity === 'Working/Studying' &&
    (location === 'Work/School' || location === 'Home')
  ) {
    return social === 'Alone'
      ? {
          title: 'Discreet Focus Reset',
          suggestion:
            'Place your hand flat on your desk. Focus on the cool, solid feeling for 60 seconds. This simple grounding technique can pull you back to the present moment without breaking your workflow.',
        }
      : {
          title: 'Silent Breathing Anchor',
          suggestion:
            'Try a subtle breathing exercise. Inhale slowly for 4 seconds, and exhale for 6. It\'s completely silent and helps calm the nervous system, even if you\'re around others.',
        };
  }
  if (activity === 'Commuting') {
    return {
      title: 'Mindful Commute',
      suggestion:
        'Put on a calming podcast or instrumental playlist. Focus on the sounds and try to loosen your grip if you\'re driving or holding onto a rail. Let the journey be a moment of transition.',
    };
  }
  if (activity === 'Socializing') {
    return {
      title: 'Grounding in the Moment',
      suggestion:
        'If you feel overwhelmed, subtly focus on one thing you can hear in your environment. Let it be an anchor. You don\'t have to leave the conversation, just find a single point of focus.',
    };
  }
  if (activity === 'Relaxing' && location === 'Home') {
    return {
      title: 'Deepen Your Relaxation',
      suggestion:
        'Since you\'re already relaxing at home, enhance it. Try a 5-minute guided meditation or simply listen to one of your favorite, most comforting songs.',
    };
  }
  return {
    title: 'A Mindful Pause',
    suggestion:
      'Take a moment to stand up, stretch your arms to the sky, and take one deep, intentional breath. A small reset can make a big difference.',
  };
};

type AppState =
  | 'IDLE'
  | 'STRESS_DETECTED'
  | 'EMA_START'
  | 'EMA_QUESTIONING'
  | 'FETCHING_SUGGESTION'
  | 'SHOWING_SUGGESTION'
  | 'VOICE_CHATBOT'
  | 'VOICE_RECORDING'
  | 'VOICE_PROCESSING'
  | 'VOICE_RESPONSE';

interface Notification {
  id: string;
  type: 'stress_detected' | 'info' | 'warning';
  message: string;
  timestamp: number;
}

interface SystemStatus {
  status: 'running' | 'offline';
  pending_notifications: number;
}

interface Suggestion {
  title: string;
  suggestion: string;
}

interface ChatMessage {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: number;
}

interface CompanionAppProps {
  embedded?: boolean;
  onClose?: () => void;
}

export default function CompanionApp({ embedded = false, onClose }: CompanionAppProps) {
  const [appState, setAppState] = useState<AppState>('IDLE');
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [emaAnswers, setEmaAnswers] = useState<Record<string, string>>({});
  const [suggestion, setSuggestion] = useState<Suggestion | null>(null);
  const [isFading, setIsFading] = useState(false);
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [stressDetected, setStressDetected] = useState(false);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [audioChunks, setAudioChunks] = useState<Blob[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [chatbotResponse, setChatbotResponse] = useState<string | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isSending, setIsSending] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement | null>(null);

  const transitionToState = (newState: AppState) => {
    setIsFading(true);
    setTimeout(() => {
      setAppState(newState);
      setIsFading(false);
    }, 300);
  };

  // --- API Functions ---
  const fetchNotifications = async () => {
    try {
      const response = await fetch('/api/notifications');
      const data = await response.json();
      setNotifications(data.notifications || []);

      // Check for stress notifications
      const stressNotifications = data.notifications.filter(
        (n: Notification) => n.type === 'stress_detected'
      );
      if (stressNotifications.length > 0 && !stressDetected) {
        setStressDetected(true);
        // Auto-trigger EMA if stress is detected
        transitionToState('STRESS_DETECTED');
      }
    } catch (error) {
      console.error('Failed to fetch notifications:', error);
    }
  };

  const fetchSystemStatus = async () => {
    try {
      const response = await fetch('/api/status');
      const data = await response.json();
      setSystemStatus(data);
    } catch (error) {
      console.error('Failed to fetch system status:', error);
    }
  };

  const acknowledgeNotification = async (notificationId: string) => {
    try {
      await fetch('/api/notifications', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'acknowledge', id: notificationId }),
      });
      await fetchNotifications(); // Refresh notifications
    } catch (error) {
      console.error('Failed to acknowledge notification:', error);
    }
  };

  const clearAllNotifications = async () => {
    try {
      await fetch('/api/notifications', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'clear' }),
      });
      setNotifications([]);
      setStressDetected(false);
    } catch (error) {
      console.error('Failed to clear notifications:', error);
    }
  };

  // --- Effects ---
  useEffect(() => {
    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }

    // Fetch initial data
    fetchNotifications();
    fetchSystemStatus();

    // Set up polling for notifications
    const interval = setInterval(() => {
      fetchNotifications();
      fetchSystemStatus();
    }, 5000); // Check every 5 seconds

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (appState === 'FETCHING_SUGGESTION') {
      setTimeout(() => {
        const result = getSuggestionFromModel(emaAnswers);
        setSuggestion(result);
        transitionToState('SHOWING_SUGGESTION');
      }, 1500);
    }
  }, [appState, emaAnswers]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (mediaRecorder && isRecording) {
        try {
          if (mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
          }
        } catch (error) {
          console.error('Error cleaning up recorder:', error);
        }
      }
    };
  }, [mediaRecorder, isRecording]);

  // Auto-scroll chat to bottom when new messages arrive
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatMessages]);

  const handleNotificationClick = () => transitionToState('EMA_START');
  const handleStartEma = () => {
    setCurrentQuestionIndex(0);
    setEmaAnswers({});
    transitionToState('EMA_QUESTIONING');
  };

  const handleStressDetected = () => {
    // Clear stress notifications and start EMA
    clearAllNotifications();
    transitionToState('EMA_START');
  };

  const handleAnswer = (questionId: string, answer: string) => {
    const newAnswers = { ...emaAnswers, [questionId]: answer };
    setEmaAnswers(newAnswers);

    if (currentQuestionIndex < EMA_QUESTIONS.length - 1) {
      setIsFading(true);
      setTimeout(() => {
        setCurrentQuestionIndex(currentQuestionIndex + 1);
        setIsFading(false);
      }, 300);
    } else {
      transitionToState('FETCHING_SUGGESTION');
    }
  };

  const handleSkip = () => transitionToState('FETCHING_SUGGESTION');
  const handleReset = () => {
    // Stop any ongoing recording
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      setIsRecording(false);
    }
    transitionToState('IDLE');
    setCurrentQuestionIndex(0);
    setEmaAnswers({});
    setSuggestion(null);
    setChatbotResponse(null);
    setMediaRecorder(null);
    setAudioChunks([]);
    setChatMessages([]);
    setInputMessage('');
  };

  const handleStartVoiceChatbot = () => {
    setChatbotResponse(null);
    setChatMessages([]);
    setInputMessage('');
    transitionToState('VOICE_CHATBOT');
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks: Blob[] = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      recorder.onstop = async () => {
        const audioBlob = new Blob(chunks, { type: 'audio/webm' });
        stream.getTracks().forEach((track) => track.stop());
        
        transitionToState('VOICE_PROCESSING');
        await processVoiceRecording(audioBlob);
      };

      recorder.start();
      setMediaRecorder(recorder);
      setAudioChunks(chunks);
      setIsRecording(true);
      transitionToState('VOICE_RECORDING');
    } catch (error) {
      console.error('Error starting recording:', error);
      alert('Failed to access microphone. Please check permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorder && isRecording) {
      try {
        if (mediaRecorder.state === 'recording') {
          mediaRecorder.stop();
        }
        setIsRecording(false);
      } catch (error) {
        console.error('Error stopping recording:', error);
        setIsRecording(false);
      }
    }
  };

  const addMessage = (text: string, sender: 'user' | 'bot') => {
    const newMessage: ChatMessage = {
      id: Date.now().toString(),
      text,
      sender,
      timestamp: Date.now(),
    };
    setChatMessages((prev) => [...prev, newMessage]);
  };

  const sendTextMessage = async (text: string) => {
    if (!text.trim() || isSending) return;

    setIsSending(true);
    const userMessageText = text.trim();
    setInputMessage('');

    try {
      // Build conversation history for context (previous messages only, before adding current)
      const history = chatMessages.map((msg) => ({
        role: msg.sender === 'user' ? 'user' : 'model',
        text: msg.text,
      }));

      // Add user message to UI immediately
      addMessage(userMessageText, 'user');

      const response = await fetch('/api/chatbot/text', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessageText,
          history: history,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get chatbot response');
      }

      const data = await response.json();
      addMessage(data.response || 'I\'m here to help. How can I assist you with managing your stress?', 'bot');
    } catch (error) {
      console.error('Error sending message:', error);
      addMessage('Sorry, I encountered an error. Please try again.', 'bot');
    } finally {
      setIsSending(false);
    }
  };

  const processVoiceRecording = async (audioBlob: Blob) => {
    try {
      // Convert audio to base64
      const reader = new FileReader();
      reader.readAsDataURL(audioBlob);
      
      reader.onloadend = async () => {
        const base64Audio = reader.result as string;
        
        // Build conversation history for context
        const history = chatMessages.map((msg) => ({
          role: msg.sender === 'user' ? 'user' : 'model',
          text: msg.text,
        }));
        
        // Send to API for transcription and chatbot response
        const response = await fetch('/api/chatbot/voice', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            audio: base64Audio,
            history: history,
          }),
        });

        if (!response.ok) {
          throw new Error('Failed to process voice recording');
        }

        const data = await response.json();
        const userMessage = data.transcription || '[Voice message]';
        const botResponse = data.response || 'I\'m here to help. How can I assist you with managing your stress?';
        
        addMessage(userMessage, 'user');
        addMessage(botResponse, 'bot');
        transitionToState('VOICE_CHATBOT');
      };
    } catch (error) {
      console.error('Error processing voice recording:', error);
      addMessage('Sorry, I encountered an error processing your voice message. Please try again.', 'bot');
      transitionToState('VOICE_CHATBOT');
    }
  };

  const simulateStressNotification = async () => {
    try {
      await fetch('/api/notifications', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'simulate_stress' }),
      });
      await fetchNotifications();
    } catch (error) {
      console.error('Failed to simulate stress notification:', error);
    }
  };

  const renderContent = () => {
    switch (appState) {
      case 'STRESS_DETECTED':
        return (
          <Card>
            <h1 className="text-lg font-semibold text-red-600 text-center mb-2 mt-0">
              🚨 Stress Detected
            </h1>
            <p className="text-sm text-slate-600 text-center mb-4 leading-relaxed">
              Our sensors have detected elevated stress levels. Let's take a moment to check in and find the best way to help you.
            </p>
            <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4">
              <p className="text-xs text-red-900 text-center m-0 leading-relaxed">
                Your physiological indicators suggest you might be experiencing stress. This is completely normal, and we're here to help.
              </p>
            </div>
            <Button title="Let's Check In" onClick={handleStressDetected} />
            <button
              onClick={handleReset}
              className="bg-transparent border-none text-slate-400 text-xs font-medium text-center cursor-pointer mt-3 px-2 py-1 hover:text-slate-600 transition-colors"
            >
              Dismiss
            </button>
          </Card>
        );
      case 'EMA_START':
        return (
          <Card>
            <h1 className="text-lg font-semibold text-slate-900 text-center mb-2 mt-0">
              Checking In
            </h1>
            <p className="text-sm text-slate-600 text-center mb-4 leading-relaxed">
              Choose how you'd like to check in. Answer a few quick questions or talk to our chatbot.
            </p>
            <Button title="Answer Questions" onClick={handleStartEma} />
            <Button title="Talk to Chatbot" onClick={handleStartVoiceChatbot} className="bg-indigo-600 hover:bg-indigo-700" />
          </Card>
        );
      case 'EMA_QUESTIONING':
        const question = EMA_QUESTIONS[currentQuestionIndex];
        return (
          <Card key={currentQuestionIndex}>
            <div className="mb-3">
              <div className="text-xs text-slate-400 mb-1 text-center">
                Question {currentQuestionIndex + 1} of {EMA_QUESTIONS.length}
              </div>
              <h2 className="text-base font-semibold text-slate-900 text-center mb-0 mt-0">
                {question.question}
              </h2>
            </div>
            <div className="w-full space-y-2">
              {question.options.map((option, index) => (
                <Button
                  key={option}
                  title={option}
                  onClick={() => handleAnswer(question.id, option)}
                  className="animate-in fade-in"
                  style={{ animationDelay: `${index * 100}ms` }}
                />
              ))}
            </div>
            <button
              onClick={handleSkip}
              className="bg-transparent border-none text-slate-400 text-xs font-medium text-center cursor-pointer mt-3 px-2 py-1 hover:text-slate-600 transition-colors"
            >
              Skip
            </button>
          </Card>
        );
      case 'FETCHING_SUGGESTION':
        return (
          <Card>
            <p className="text-sm text-slate-600 text-center mb-4">Analyzing your context...</p>
            <div className="border-2 border-slate-200 border-t-indigo-600 rounded-full w-8 h-8 animate-spin mx-auto" />
          </Card>
        );
      case 'SHOWING_SUGGESTION':
        return (
          <Card>
            <h1 className="text-base font-semibold text-slate-900 text-center mb-2 mt-0">
              {suggestion?.title}
            </h1>
            <p className="text-sm text-slate-700 text-center mb-4 leading-relaxed">
              {suggestion?.suggestion}
            </p>
            <Button title="Done" onClick={handleReset} />
          </Card>
        );
      case 'VOICE_CHATBOT':
        return (
          <Card>
            <div className="w-full flex items-center justify-between mb-3">
              <h1 className="text-base font-semibold text-slate-900 m-0">
                Chat with Bot
              </h1>
              <button
                onClick={() => {
                  setChatMessages([]);
                  transitionToState('EMA_START');
                }}
                className="bg-transparent border-none text-slate-400 text-xs font-medium cursor-pointer px-2 py-1 hover:text-slate-600 transition-colors"
              >
                Back
              </button>
            </div>
            
            {/* Chat Messages */}
            <div 
              ref={chatContainerRef}
              className="w-full h-56 mb-3 overflow-y-auto border border-slate-200 rounded-lg p-3 bg-slate-50/50 flex flex-col gap-2 scrollbar-thin"
            >
              {chatMessages.length === 0 ? (
                <p className="text-xs text-slate-400 text-center mt-2">
                  Start a conversation by typing a message or using voice
                </p>
              ) : (
                chatMessages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[85%] rounded-lg px-3 py-2 ${
                        msg.sender === 'user'
                          ? 'bg-indigo-600 text-white'
                          : 'bg-white text-slate-800 border border-slate-200 shadow-sm'
                      }`}
                    >
                      <p className="text-xs m-0 leading-relaxed">{msg.text}</p>
                    </div>
                  </div>
                ))
              )}
              {isSending && (
                <div className="flex justify-start">
                  <div className="bg-white text-slate-800 border border-slate-200 rounded-lg px-3 py-2 shadow-sm">
                    <div className="flex items-center gap-1.5">
                      <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                      <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                      <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Text Input */}
            <div className="w-full flex gap-2 mb-3">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendTextMessage(inputMessage);
                  }
                }}
                placeholder="Type your message..."
                disabled={isSending}
                className="flex-1 px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-sm"
              />
              <button
                onClick={() => sendTextMessage(inputMessage)}
                disabled={!inputMessage.trim() || isSending}
                className="px-3 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-xs font-medium"
              >
                Send
              </button>
            </div>

            {/* Voice Button */}
            <div className="flex flex-col items-center">
              <p className="text-xs text-slate-400 mb-1.5">Or press and hold to talk</p>
              <VoiceButton
                onPressStart={startRecording}
                onPressEnd={stopRecording}
                isRecording={isRecording}
              />
            </div>
          </Card>
        );
      case 'VOICE_RECORDING':
        return (
          <Card>
            <div className="w-full flex items-center justify-between mb-3">
              <h1 className="text-base font-semibold text-slate-900 m-0">
                Listening...
              </h1>
              <button
                onClick={() => {
                  if (isRecording) stopRecording();
                  transitionToState('VOICE_CHATBOT');
                }}
                className="bg-transparent border-none text-slate-400 text-xs font-medium cursor-pointer px-2 py-1 hover:text-slate-600 transition-colors"
              >
                Cancel
              </button>
            </div>
            
            {/* Show chat messages while recording */}
            <div className="w-full h-40 mb-3 overflow-y-auto border border-slate-200 rounded-lg p-3 bg-slate-50/50 flex flex-col gap-2">
              {chatMessages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[85%] rounded-lg px-3 py-2 ${
                      msg.sender === 'user'
                        ? 'bg-indigo-600 text-white'
                        : 'bg-white text-slate-800 border border-slate-200 shadow-sm'
                    }`}
                  >
                    <p className="text-xs m-0 leading-relaxed">{msg.text}</p>
                  </div>
                </div>
              ))}
            </div>

            <p className="text-xs text-slate-600 text-center mb-3 leading-relaxed">
              Keep holding the button and speak. Release when finished.
            </p>
            <VoiceButton
              onPressStart={startRecording}
              onPressEnd={stopRecording}
              isRecording={true}
            />
            <div className="mt-3 flex items-center justify-center gap-2">
              <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
              <span className="text-xs text-slate-600">Recording</span>
            </div>
          </Card>
        );
      case 'VOICE_PROCESSING':
        return (
          <Card>
            <div className="w-full flex items-center justify-between mb-3">
              <h1 className="text-base font-semibold text-slate-900 m-0">
                Processing...
              </h1>
            </div>
            
            {/* Show chat messages while processing */}
            <div className="w-full h-40 mb-3 overflow-y-auto border border-slate-200 rounded-lg p-3 bg-slate-50/50 flex flex-col gap-2">
              {chatMessages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[85%] rounded-lg px-3 py-2 ${
                      msg.sender === 'user'
                        ? 'bg-indigo-600 text-white'
                        : 'bg-white text-slate-800 border border-slate-200 shadow-sm'
                    }`}
                  >
                    <p className="text-xs m-0 leading-relaxed">{msg.text}</p>
                  </div>
                </div>
              ))}
              <div className="flex justify-start">
                <div className="bg-white text-slate-800 border border-slate-200 rounded-lg px-3 py-2 shadow-sm">
                  <div className="flex items-center gap-1.5">
                    <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              </div>
            </div>

            <p className="text-xs text-slate-600 text-center mb-3">Understanding what you said...</p>
            <div className="border-2 border-slate-200 border-t-indigo-600 rounded-full w-8 h-8 animate-spin mx-auto" />
          </Card>
        );
      case 'IDLE':
      default:
        return (
          <Card>
            <h1 className="text-base font-semibold text-slate-900 text-center mb-2 mt-0">
              Mindwell Companion
            </h1>
            <p className="text-xs text-slate-600 text-center mb-4 leading-relaxed">
              Your personal guide to moments of calm, triggered by your wearable when you need it most.
            </p>

            {/* System Status */}
            {systemStatus && (
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-2.5 mb-3">
                <p className="text-xs text-slate-600 m-0 text-center">
                  Status:{' '}
                  {systemStatus.status === 'running' ? '🟢 Monitoring' : '🔴 Offline'}
                </p>
                {systemStatus.pending_notifications > 0 && (
                  <p className="text-xs text-red-600 m-0 mt-1 text-center font-medium">
                    {systemStatus.pending_notifications} notification(s) pending
                  </p>
                )}
              </div>
            )}

            {/* Notifications */}
            {notifications.length > 0 && (
              <div className="bg-amber-50 border border-amber-300 rounded-lg p-3 mb-3">
                <h3 className="text-xs font-semibold text-amber-900 m-0 mb-2 text-center">
                  Recent Alerts
                </h3>
                {notifications.slice(0, 3).map((notification) => (
                  <div
                    key={notification.id}
                    className="bg-white border border-slate-200 rounded-lg p-2 mb-1.5 flex justify-between items-center last:mb-0"
                  >
                    <p className="text-xs text-slate-700 m-0 flex-1 leading-snug">
                      {notification.message}
                    </p>
                    <button
                      onClick={() => acknowledgeNotification(notification.id)}
                      className="bg-slate-500 text-white border-none rounded px-2 py-1 text-xs cursor-pointer ml-2 hover:bg-slate-600 transition-colors"
                    >
                      Dismiss
                    </button>
                  </div>
                ))}
              </div>
            )}

            <Button title="Simulate Stress Notification" onClick={simulateStressNotification} />
            {notifications.length > 0 && (
              <Button
                title="Clear All Notifications"
                onClick={clearAllNotifications}
                className="bg-red-600 hover:bg-red-700"
              />
            )}
          </Card>
        );
    }
  };

  if (embedded) {
    return (
      <div className="flex justify-center items-center bg-slate-100 p-4 box-border relative min-h-full w-full">
        <div className={`transition-opacity duration-300 w-full ${isFading ? 'opacity-0' : 'opacity-100'}`}>
          {renderContent()}
        </div>
      </div>
    );
  }

  return (
    <main className="flex min-h-screen justify-center items-center bg-slate-100 p-5 box-border relative">
      <Link
        href="/"
        className="absolute top-8 left-8 flex items-center gap-2 text-slate-600 hover:text-slate-900 transition-colors text-sm font-medium"
      >
        <ArrowLeft className="w-4 h-4" />
        Back to Dashboard
      </Link>
      <div className={`transition-opacity duration-300 ${isFading ? 'opacity-0' : 'opacity-100'}`}>
        {renderContent()}
      </div>
    </main>
  );
}

// --- Reusable Components ---
const Card = ({ children }: { children: React.ReactNode }) => (
  <div className="w-full max-w-[420px] bg-white rounded-2xl p-5 flex flex-col border border-slate-200/60 shadow-sm box-border mx-auto">
    {children}
  </div>
);

const Button = ({
  title,
  onClick,
  className = '',
  style,
}: {
  title: string;
  onClick: () => void;
  className?: string;
  style?: React.CSSProperties;
}) => (
  <button
    className={`w-full bg-slate-900 py-2.5 px-4 rounded-lg mt-2 text-white text-center text-sm font-medium border-none cursor-pointer shadow-sm transition-all duration-200 hover:bg-slate-800 active:scale-[0.98] ${className}`}
    onClick={onClick}
    style={style}
  >
    {title}
  </button>
);

const VoiceButton = ({
  onPressStart,
  onPressEnd,
  isRecording,
}: {
  onPressStart: () => void;
  onPressEnd: () => void;
  isRecording: boolean;
}) => {
  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    onPressStart();
  };

  const handleMouseUp = (e: React.MouseEvent) => {
    e.preventDefault();
    onPressEnd();
  };

  const handleMouseLeave = (e: React.MouseEvent) => {
    e.preventDefault();
    if (isRecording) {
      onPressEnd();
    }
  };

  const handleTouchStart = (e: React.TouchEvent) => {
    e.preventDefault();
    onPressStart();
  };

  const handleTouchEnd = (e: React.TouchEvent) => {
    e.preventDefault();
    onPressEnd();
  };

  return (
    <button
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseLeave}
      onTouchStart={handleTouchStart}
      onTouchEnd={handleTouchEnd}
      className={`
        w-20 h-20 rounded-full border-none cursor-pointer shadow-md
        transition-all duration-200 flex items-center justify-center
        ${isRecording
          ? 'bg-red-500 hover:bg-red-600 scale-105 animate-pulse'
          : 'bg-indigo-600 hover:bg-indigo-700 active:scale-95'
        }
      `}
      style={{ touchAction: 'none' }}
    >
      <svg
        className="w-8 h-8 text-white"
        fill="currentColor"
        viewBox="0 0 24 24"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
        <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
      </svg>
    </button>
  );
};


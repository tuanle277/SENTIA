import React, { useState, useEffect } from "react";

// --- Configuration ---
const EMA_QUESTIONS = [
  {
    id: "activity",
    question: "What are you primarily doing right now?",
    options: [
      "Working/Studying",
      "Commuting",
      "Socializing",
      "Relaxing",
      "Chores",
    ],
  },
  {
    id: "social",
    question: "Who are you with?",
    options: ["Alone", "With Partner/Family", "With Friends", "With Coworkers"],
  },
  {
    id: "location",
    question: "How would you describe your environment?",
    options: ["Home", "Work/School", "Public Space", "Outdoors"],
  },
];

// --- Mock Backend/Model Logic ---
const getSuggestionFromModel = (context) => {
  const { activity, social, location } = context;
  if (
    activity === "Working/Studying" &&
    (location === "Work/School" || location === "Home")
  ) {
    return social === "Alone"
      ? {
          title: "Discreet Focus Reset",
          suggestion:
            "Place your hand flat on your desk. Focus on the cool, solid feeling for 60 seconds. This simple grounding technique can pull you back to the present moment without breaking your workflow.",
        }
      : {
          title: "Silent Breathing Anchor",
          suggestion:
            "Try a subtle breathing exercise. Inhale slowly for 4 seconds, and exhale for 6. It's completely silent and helps calm the nervous system, even if you're around others.",
        };
  }
  if (activity === "Commuting") {
    return {
      title: "Mindful Commute",
      suggestion:
        "Put on a calming podcast or instrumental playlist. Focus on the sounds and try to loosen your grip if you're driving or holding onto a rail. Let the journey be a moment of transition.",
    };
  }
  if (activity === "Socializing") {
    return {
      title: "Grounding in the Moment",
      suggestion:
        "If you feel overwhelmed, subtly focus on one thing you can hear in your environment. Let it be an anchor. You don't have to leave the conversation, just find a single point of focus.",
    };
  }
  if (activity === "Relaxing" && location === "Home") {
    return {
      title: "Deepen Your Relaxation",
      suggestion:
        "Since you're already relaxing at home, enhance it. Try a 5-minute guided meditation or simply listen to one of your favorite, most comforting songs.",
    };
  }
  return {
    title: "A Mindful Pause",
    suggestion:
      "Take a moment to stand up, stretch your arms to the sky, and take one deep, intentional breath. A small reset can make a big difference.",
  };
};

// --- API Configuration ---
const API_BASE_URL = "http://localhost:5000/api";

// --- Main App Component ---
export default function App() {
  const [appState, setAppState] = useState("IDLE");
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [emaAnswers, setEmaAnswers] = useState({});
  const [suggestion, setSuggestion] = useState(null);
  const [isFading, setIsFading] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [stressDetected, setStressDetected] = useState(false);
  const [systemStatus, setSystemStatus] = useState(null);

  const transitionToState = (newState) => {
    setIsFading(true);
    setTimeout(() => {
      setAppState(newState);
      setIsFading(false);
    }, 300); // This delay should match the CSS transition duration
  };

  // --- API Functions ---
  const fetchNotifications = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/notifications`);
      const data = await response.json();
      setNotifications(data.notifications || []);

      // Check for stress notifications
      const stressNotifications = data.notifications.filter(
        (n) => n.type === "stress_detected"
      );
      if (stressNotifications.length > 0 && !stressDetected) {
        setStressDetected(true);
        // Auto-trigger EMA if stress is detected
        transitionToState("STRESS_DETECTED");
      }
    } catch (error) {
      console.error("Failed to fetch notifications:", error);
    }
  };

  const fetchSystemStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/status`);
      const data = await response.json();
      setSystemStatus(data);
    } catch (error) {
      console.error("Failed to fetch system status:", error);
    }
  };

  const acknowledgeNotification = async (notificationId) => {
    try {
      await fetch(`${API_BASE_URL}/notifications/acknowledge`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: notificationId }),
      });
      await fetchNotifications(); // Refresh notifications
    } catch (error) {
      console.error("Failed to acknowledge notification:", error);
    }
  };

  const clearAllNotifications = async () => {
    try {
      await fetch(`${API_BASE_URL}/notifications/clear`, { method: "POST" });
      setNotifications([]);
      setStressDetected(false);
    } catch (error) {
      console.error("Failed to clear notifications:", error);
    }
  };

  // --- Effects ---
  useEffect(() => {
    // Request notification permission
    if ("Notification" in window && Notification.permission === "default") {
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
    if (appState === "FETCHING_SUGGESTION") {
      setTimeout(() => {
        const result = getSuggestionFromModel(emaAnswers);
        setSuggestion(result);
        transitionToState("SHOWING_SUGGESTION");
      }, 1500);
    }
  }, [appState, emaAnswers]);

  const handleNotificationClick = () => transitionToState("EMA_START");
  const handleStartEma = () => {
    setCurrentQuestionIndex(0);
    setEmaAnswers({});
    transitionToState("EMA_QUESTIONING");
  };

  const handleStressDetected = () => {
    // Clear stress notifications and start EMA
    clearAllNotifications();
    transitionToState("EMA_START");
  };

  const handleAnswer = (questionId, answer) => {
    const newAnswers = { ...emaAnswers, [questionId]: answer };
    setEmaAnswers(newAnswers);

    if (currentQuestionIndex < EMA_QUESTIONS.length - 1) {
      setIsFading(true);
      setTimeout(() => {
        setCurrentQuestionIndex(currentQuestionIndex + 1);
        setIsFading(false);
      }, 300);
    } else {
      transitionToState("FETCHING_SUGGESTION");
    }
  };

  const handleSkip = () => transitionToState("FETCHING_SUGGESTION");
  const handleReset = () => {
    transitionToState("IDLE");
    setCurrentQuestionIndex(0);
    setEmaAnswers({});
    setSuggestion(null);
  };

  const renderContent = () => {
    switch (appState) {
      case "STRESS_DETECTED":
        return (
          <Card>
            <h1 style={styles.stressTitle}>🚨 Stress Detected</h1>
            <p style={styles.subtitle}>
              Our sensors have detected elevated stress levels. Let's take a
              moment to check in and find the best way to help you.
            </p>
            <div style={styles.stressInfo}>
              <p style={styles.stressText}>
                Your physiological indicators suggest you might be experiencing
                stress. This is completely normal, and we're here to help.
              </p>
            </div>
            <Button title="Let's Check In" onClick={handleStressDetected} />
            <button
              onClick={handleReset}
              className="skip-button"
              style={styles.skipButton}
            >
              Dismiss
            </button>
          </Card>
        );
      case "EMA_START":
        return (
          <Card>
            <h1 style={styles.title}>Checking In</h1>
            <p style={styles.subtitle}>
              Let's find the best way to help. A few quick questions will find a
              personalized suggestion for you.
            </p>
            <Button title="Begin" onClick={handleStartEma} />
          </Card>
        );
      case "EMA_QUESTIONING":
        const question = EMA_QUESTIONS[currentQuestionIndex];
        return (
          <Card key={currentQuestionIndex}>
            <h2 style={styles.title}>{question.question}</h2>
            <div style={{ width: "100%" }}>
              {question.options.map((option, index) => (
                <Button
                  key={option}
                  title={option}
                  onClick={() => handleAnswer(question.id, option)}
                  style={{ animationDelay: `${index * 100}ms` }}
                  className="button-animate-in"
                />
              ))}
            </div>
            <button
              onClick={handleSkip}
              className="skip-button"
              style={styles.skipButton}
            >
              Skip
            </button>
          </Card>
        );
      case "FETCHING_SUGGESTION":
        return (
          <Card>
            <p style={styles.subtitle}>Analyzing your context...</p>
            <div style={styles.spinner}></div>
          </Card>
        );
      case "SHOWING_SUGGESTION":
        return (
          <Card>
            <h1 style={styles.title}>{suggestion.title}</h1>
            <p style={styles.suggestionText}>{suggestion.suggestion}</p>
            <Button title="Done" onClick={handleReset} />
          </Card>
        );
      case "IDLE":
      default:
        return (
          <Card>
            <h1 style={styles.title}>Mindwell Companion</h1>
            <p style={styles.subtitle}>
              Your personal guide to moments of calm, triggered by your wearable
              when you need it most.
            </p>

            {/* System Status */}
            {systemStatus && (
              <div style={styles.statusContainer}>
                <p style={styles.statusText}>
                  Status:{" "}
                  {systemStatus.status === "running"
                    ? "🟢 Monitoring"
                    : "🔴 Offline"}
                </p>
                {systemStatus.pending_notifications > 0 && (
                  <p style={styles.notificationCount}>
                    {systemStatus.pending_notifications} notification(s) pending
                  </p>
                )}
              </div>
            )}

            {/* Notifications */}
            {notifications.length > 0 && (
              <div style={styles.notificationsContainer}>
                <h3 style={styles.notificationsTitle}>Recent Alerts</h3>
                {notifications.slice(0, 3).map((notification, index) => (
                  <div key={notification.id} style={styles.notificationItem}>
                    <p style={styles.notificationText}>
                      {notification.message}
                    </p>
                    <button
                      onClick={() => acknowledgeNotification(notification.id)}
                      style={styles.dismissButton}
                    >
                      Dismiss
                    </button>
                  </div>
                ))}
              </div>
            )}

            <div style={{ marginTop: 20 }} />
            <Button
              title="Simulate Stress Notification"
              onClick={handleNotificationClick}
            />
            {notifications.length > 0 && (
              <Button
                title="Clear All Notifications"
                onClick={clearAllNotifications}
                style={styles.clearButton}
              />
            )}
          </Card>
        );
    }
  };

  return (
    <main style={styles.container}>
      <style>{`
        body { 
          margin: 0; 
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes fadeOut { from { opacity: 1; transform: translateY(0); } to { opacity: 0; transform: translateY(-10px); } }
        
        .card-content {
          animation: fadeIn 0.3s ease-out forwards;
        }
        .card-content.fading {
          animation: fadeOut 0.3s ease-in forwards;
        }
        
        .button-animate-in {
          animation: fadeIn 0.5s ease-out forwards;
          opacity: 0; /* Start hidden */
        }

        .mindwell-button {
          transition: background-color 0.2s ease, transform 0.1s ease;
        }
        .mindwell-button:hover {
          background-color: #262626 !important; /* A darker gray for hover */
          transform: translateY(-2px);
        }
        .mindwell-button:active {
          transform: translateY(0px);
        }
        .skip-button {
          transition: color 0.2s ease;
        }
        .skip-button:hover {
          color: #525252 !important;
        }
      `}</style>
      <div className={`card-content ${isFading ? "fading" : ""}`}>
        {renderContent()}
      </div>
    </main>
  );
}

// --- Reusable Components & Styles ---
const Card = ({ children, key }) => (
  <div key={key} style={styles.card}>
    {children}
  </div>
);
const Button = ({ title, onClick, style, className }) => (
  <button
    className={`mindwell-button ${className || ""}`}
    style={{ ...styles.button, ...style }}
    onClick={onClick}
  >
    {title}
  </button>
);

const styles = {
  container: {
    display: "flex",
    minHeight: "100vh",
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#f5f5f5", // A calm, neutral gray
    padding: 20,
    boxSizing: "border-box",
  },
  card: {
    width: "100%",
    maxWidth: 380,
    backgroundColor: "#ffffff",
    borderRadius: 24,
    padding: "32px 28px",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    border: "1px solid #e5e5e5",
    boxShadow:
      "0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -2px rgba(0, 0, 0, 0.03)",
    boxSizing: "border-box",
  },
  title: {
    fontSize: 22,
    fontWeight: "700",
    color: "#171717", // Near-black for high contrast
    textAlign: "center",
    marginBottom: 12,
    marginTop: 0,
  },
  subtitle: {
    fontSize: 15,
    color: "#737373", // A soft, secondary gray
    textAlign: "center",
    marginBottom: 28,
    lineHeight: 1.6,
    marginTop: 0,
    maxWidth: "90%",
  },
  suggestionText: {
    fontSize: 16,
    color: "#404040",
    textAlign: "center",
    marginBottom: 28,
    lineHeight: 1.7,
    marginTop: 0,
  },
  button: {
    width: "100%",
    backgroundColor: "#171717", // Matches the title color for a cohesive look
    padding: "15px 0",
    borderRadius: 14,
    marginTop: 10,
    color: "#ffffff",
    textAlign: "center",
    fontSize: 15,
    fontWeight: "500",
    border: "none",
    cursor: "pointer",
    boxShadow: "0 4px 14px 0 rgba(0, 0, 0, 0.05)",
  },
  skipButton: {
    background: "none",
    border: "none",
    color: "#a3a3a3",
    fontSize: 14,
    fontWeight: "500",
    textAlign: "center",
    cursor: "pointer",
    marginTop: 16,
    padding: 8,
  },
  spinner: {
    border: "4px solid #e5e5e5",
    borderTop: "4px solid #404040",
    borderRadius: "50%",
    width: 36,
    height: 36,
    animation: "spin 1s linear infinite",
    margin: "20px 0",
  },
  stressTitle: {
    fontSize: 24,
    fontWeight: "700",
    color: "#dc2626", // Red for stress
    textAlign: "center",
    marginBottom: 12,
    marginTop: 0,
  },
  stressInfo: {
    backgroundColor: "#fef2f2",
    border: "1px solid #fecaca",
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
  },
  stressText: {
    fontSize: 14,
    color: "#991b1b",
    textAlign: "center",
    margin: 0,
    lineHeight: 1.5,
  },
  statusContainer: {
    backgroundColor: "#f8fafc",
    border: "1px solid #e2e8f0",
    borderRadius: 12,
    padding: 12,
    marginBottom: 16,
  },
  statusText: {
    fontSize: 14,
    color: "#475569",
    margin: 0,
    textAlign: "center",
  },
  notificationCount: {
    fontSize: 12,
    color: "#dc2626",
    margin: "4px 0 0 0",
    textAlign: "center",
    fontWeight: "500",
  },
  notificationsContainer: {
    backgroundColor: "#fef3c7",
    border: "1px solid #fbbf24",
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  notificationsTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: "#92400e",
    margin: "0 0 12px 0",
    textAlign: "center",
  },
  notificationItem: {
    backgroundColor: "#ffffff",
    border: "1px solid #f3f4f6",
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  notificationText: {
    fontSize: 13,
    color: "#374151",
    margin: 0,
    flex: 1,
    lineHeight: 1.4,
  },
  dismissButton: {
    backgroundColor: "#6b7280",
    color: "#ffffff",
    border: "none",
    borderRadius: 6,
    padding: "4px 8px",
    fontSize: 12,
    cursor: "pointer",
    marginLeft: 8,
  },
  clearButton: {
    backgroundColor: "#dc2626",
    marginTop: 8,
  },
};

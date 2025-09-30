#!/usr/bin/env python3
"""
Main script to run the complete stress monitoring system.

This script starts:
1. The stress detection service (monitors physiological data)
2. The notification system (handles alerts and communication)
3. Integration between the two systems

The React app should be running separately on port 3000.
"""

import os
import sys
import time
import signal
import logging
import threading
from datetime import datetime
import json

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stress_detector_service import StressDetectorService
from notification_system import NotificationSystem, StressNotificationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('stress_monitoring.log')
    ]
)
logger = logging.getLogger(__name__)

class StressMonitoringSystem:
    """
    Main system that coordinates stress detection and notifications.
    """
    
    def __init__(self):
        self.manager = None
        self.is_running = False
        self.display_thread = None
        self.latest_measurements = {}
        self.measurement_history = []
        self.max_history = 20  # Keep last 20 measurements
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("StressMonitoringSystem initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def _update_measurements(self, sensor_data, prediction=None):
        """Update measurement data for display."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        measurement = {
            'timestamp': timestamp,
            'eda_mean': sensor_data.get('EDA_Mean', 0),
            'scr_peaks': sensor_data.get('SCR_Peaks_N', 0),
            'acc_mag': sensor_data.get('ACC_Mag_Mean', 0),
            'temp_mean': sensor_data.get('TEMP_Mean', 0),
            'prediction': prediction
        }
        
        self.latest_measurements = measurement
        self.measurement_history.append(measurement)
        
        # Keep only recent history
        if len(self.measurement_history) > self.max_history:
            self.measurement_history.pop(0)
    
    def _display_measurements(self):
        """Display real-time measurements in a formatted way."""
        while self.is_running:
            try:
                # Clear screen (works on most terminals)
                os.system('clear' if os.name == 'posix' else 'cls')
                
                print("🧠 Mindwell Stress Monitoring System - Live Dashboard")
                print("=" * 80)
                print(f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"📊 Status: {'🟢 Monitoring' if self.is_running else '🔴 Stopped'}")
                print()
                
                if self.latest_measurements:
                    m = self.latest_measurements
                    print("📈 CURRENT MEASUREMENTS")
                    print("-" * 40)
                    print(f"🫀 EDA (Electrodermal Activity): {m['eda_mean']:.2f} μS")
                    print(f"⚡ SCR Peaks (Stress Response): {m['scr_peaks']} peaks")
                    print(f"🏃 Activity Level: {m['acc_mag']:.2f} g")
                    print(f"🌡️  Skin Temperature: {m['temp_mean']:.1f}°C")
                    
                    if m['prediction']:
                        pred = m['prediction']
                        stress_level = "🔴 HIGH STRESS" if pred['state'] == 'stress' else "🟢 NORMAL"
                        confidence = pred['confidence']
                        print(f"🧠 Stress Level: {stress_level}")
                        print(f"📊 Confidence: {confidence:.1%}")
                        
                        if pred['state'] == 'stress':
                            print("⚠️  STRESS DETECTED - Notification sent!")
                    else:
                        print("🧠 Stress Level: 🔄 Analyzing...")
                    
                    print()
                    
                    # Show recent history
                    print("📊 RECENT MEASUREMENTS (Last 10)")
                    print("-" * 40)
                    recent = self.measurement_history[-10:] if len(self.measurement_history) >= 10 else self.measurement_history
                    
                    for i, hist in enumerate(recent):
                        stress_indicator = "🔴" if hist['prediction'] and hist['prediction']['state'] == 'stress' else "🟢"
                        print(f"{hist['timestamp']} {stress_indicator} EDA:{hist['eda_mean']:.1f} "
                              f"SCR:{hist['scr_peaks']} ACC:{hist['acc_mag']:.1f} TEMP:{hist['temp_mean']:.1f}°C")
                    
                    print()
                    
                    # Show trends
                    if len(self.measurement_history) >= 3:
                        print("📈 TRENDS")
                        print("-" * 40)
                        recent_eda = [h['eda_mean'] for h in self.measurement_history[-3:]]
                        recent_scr = [h['scr_peaks'] for h in self.measurement_history[-3:]]
                        
                        eda_trend = "📈 Rising" if recent_eda[-1] > recent_eda[0] else "📉 Falling" if recent_eda[-1] < recent_eda[0] else "➡️ Stable"
                        scr_trend = "📈 Rising" if recent_scr[-1] > recent_scr[0] else "📉 Falling" if recent_scr[-1] < recent_scr[0] else "➡️ Stable"
                        
                        print(f"🫀 EDA Trend: {eda_trend}")
                        print(f"⚡ SCR Trend: {scr_trend}")
                        
                        # Stress detection frequency
                        stress_count = sum(1 for h in self.measurement_history if h['prediction'] and h['prediction']['state'] == 'stress')
                        total_measurements = len(self.measurement_history)
                        if total_measurements > 0:
                            stress_rate = (stress_count / total_measurements) * 100
                            print(f"🚨 Stress Detection Rate: {stress_rate:.1f}% ({stress_count}/{total_measurements})")
                    
                    print()
                else:
                    print("⏳ Waiting for first measurement...")
                    print("   The system is initializing and will start collecting data shortly.")
                
                print("💡 Tips:")
                print("   • EDA > 6.0 μS typically indicates stress")
                print("   • SCR Peaks > 10 suggest high arousal")
                print("   • Low temperature (< 32°C) can indicate stress")
                print("   • High activity with low EDA suggests physical exertion")
                print()
                print("Press Ctrl+C to stop monitoring")
                
                time.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                logger.error(f"Error in display thread: {e}")
                time.sleep(1)
    
    def _start_display_thread(self):
        """Start the measurement display thread."""
        if not self.display_thread or not self.display_thread.is_alive():
            self.display_thread = threading.Thread(target=self._display_measurements, daemon=True)
            self.display_thread.start()
            logger.info("📊 Measurement display started")
    
    def start(self):
        """Start the complete stress monitoring system."""
        try:
            # Setup paths
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, 'models', 'stress_model_rf.pkl')
            actions_path = os.path.join(current_dir, 'data', 'knowledge_base', 'actions.json')
            explanations_path = os.path.join(current_dir, 'data', 'knowledge_base', 'explanations.json')
            
            # Check if required files exist
            if not os.path.exists(model_path):
                logger.error(f"❌ Model file not found: {model_path}")
                logger.error("Please ensure you have trained the model first by running train_.py")
                return False
            
            if not os.path.exists(actions_path):
                logger.error(f"❌ Actions file not found: {actions_path}")
                return False
                
            if not os.path.exists(explanations_path):
                logger.error(f"❌ Explanations file not found: {explanations_path}")
                return False
            
            # Initialize services
            logger.info("🔧 Initializing stress detection service...")
            stress_service = StressDetectorService(model_path, actions_path, explanations_path)
            
            logger.info("🔧 Initializing notification system...")
            notification_system = NotificationSystem(react_app_port=5173)
            
            # Create manager
            logger.info("🔧 Creating stress notification manager...")
            self.manager = StressNotificationManager(stress_service, notification_system)
            
            # Set up custom callbacks to capture measurement data
            def on_stress_detected(prediction, features, suggestion):
                # Update measurements with prediction
                self._update_measurements(features, prediction)
                logger.warning(f"🚨 STRESS DETECTED! Confidence: {prediction['confidence']:.2f}")
                logger.info(f"💡 Suggestion: {suggestion}")
            
            def on_notification_needed(prediction, suggestion):
                logger.info(f"📱 NOTIFICATION: {suggestion}")
            
            # Override the manager's callbacks
            stress_service.set_stress_callback(on_stress_detected)
            stress_service.set_notification_callback(on_notification_needed)
            
            # Add measurement callback to stress service
            original_monitoring_loop = stress_service._monitoring_loop
            
            def enhanced_monitoring_loop():
                while stress_service.is_running:
                    try:
                        # Simulate getting sensor data
                        sensor_data = stress_service._simulate_sensor_data()
                        
                        if sensor_data:
                            # Extract features from sensor data
                            features = stress_service._extract_features(sensor_data)
                            
                            if features:
                                # Make prediction
                                prediction = stress_service._predict_stress(features)
                                
                                # Update measurements (always, not just on stress)
                                self._update_measurements(sensor_data, prediction)
                                
                                if prediction and prediction['state'] == 'stress':
                                    stress_service._handle_stress_detection(prediction, features)
                        
                        # Wait before next detection
                        time.sleep(stress_service.detection_interval)
                        
                    except Exception as e:
                        logger.error(f"Error in monitoring loop: {e}")
                        time.sleep(5)
            
            # Replace the monitoring loop
            stress_service._monitoring_loop = enhanced_monitoring_loop
            
            # Start monitoring
            logger.info("🚀 Starting stress monitoring system...")
            self.manager.start_monitoring()
            self.is_running = True
            
            # Start the measurement display
            self._start_display_thread()
            
            logger.info("✅ Stress monitoring system is now running!")
            logger.info("📱 React app should be running on http://localhost:5173")
            logger.info("🔔 Notification API running on http://localhost:5000")
            logger.info("📊 Live measurements will be displayed in real-time")
            logger.info("⏹️  Press Ctrl+C to stop")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start system: {e}")
            return False
    
    def stop(self):
        """Stop the stress monitoring system."""
        if self.manager and self.is_running:
            logger.info("⏹️ Stopping stress monitoring system...")
            self.is_running = False  # Stop display thread first
            self.manager.stop_monitoring()
            
            # Wait for display thread to finish
            if self.display_thread and self.display_thread.is_alive():
                self.display_thread.join(timeout=2)
            
            logger.info("✅ System stopped")
    
    def get_status(self):
        """Get system status."""
        if self.manager:
            return self.manager.get_status()
        return {"status": "not_initialized"}
    
    def run_interactive(self):
        """Run the system with interactive commands."""
        if not self.start():
            return
        
        try:
            while self.is_running:
                print("\n" + "="*50)
                print("STRESS MONITORING SYSTEM - Interactive Mode")
                print("="*50)
                print("Commands:")
                print("  status  - Show system status")
                print("  test    - Simulate stress detection")
                print("  quit    - Stop the system")
                print("="*50)
                
                command = input("\nEnter command: ").strip().lower()
                
                if command == 'status':
                    status = self.get_status()
                    print(f"\n📊 System Status:")
                    print(f"  Stress Detection: {'🟢 Running' if status['stress_detection']['is_running'] else '🔴 Stopped'}")
                    print(f"  Last Prediction: {status['stress_detection']['last_prediction_time'] or 'None'}")
                    print(f"  Pending Notifications: {status['notifications']['pending_notifications']}")
                    print(f"  Last Notification: {status['notifications']['last_notification'] or 'None'}")
                
                elif command == 'test':
                    print("\n🧪 Simulating stress detection...")
                    # This would trigger a test stress detection
                    # For now, just show a message
                    print("Test stress detection would be triggered here")
                
                elif command == 'quit':
                    break
                
                else:
                    print("❌ Unknown command. Try 'status', 'test', or 'quit'")
        
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def main():
    """Main function."""
    print("🧠 Mindwell Stress Monitoring System")
    print("="*50)
    
    # Check if we're in the right directory
    if not os.path.exists('models') or not os.path.exists('data'):
        print("❌ Error: Please run this script from the project root directory")
        print("   (the directory containing 'models' and 'data' folders)")
        sys.exit(1)
    
    # Create and run the system
    system = StressMonitoringSystem()
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        system.run_interactive()
    else:
        # Run with live measurement display
        if system.start():
            try:
                # The display thread will handle the UI
                # Just keep the main thread alive
                while system.is_running:
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print("\n⏹️ Shutting down...")
                system.stop()
        else:
            print("❌ Failed to start the system")
            sys.exit(1)


if __name__ == "__main__":
    main()

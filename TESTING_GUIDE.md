# EMA App Testing Guide

## Quick Start Testing

### 1. Start the Development Server

```bash
cd app
npm install
npm run dev
```

### 2. Open the App

Navigate to `http://localhost:5173` in your browser.

## Testing Scenarios

### 🧪 Built-in Testing Utilities

The app includes testing utilities on the main screen:

#### Direct Activity Testing

- **Test Breathing Exercise** - Launches 5-minute guided breathing
- **Test Stretching** - Launches 3-minute gentle stretching
- **Test Meditation** - Launches 2-minute mindful pause
- **Test Movement** - Launches 1-minute energy boost

#### Scenario Testing

- **Test High Stress Scenario** - Simulates extremely stressed user
- **Test Low Energy Scenario** - Simulates low energy + unfocused user

### 📋 Manual Testing Workflows

#### 1. Complete EMA Flow Testing

**Test Path 1: High Stress User**

1. Click "Simulate Stress Notification"
2. Answer EMA questions with these responses:
   - Activity: "Working/Studying"
   - Social: "Alone"
   - Location: "Work/School"
   - Stress Level: "Extremely stressed"
   - Mood: "Very negative"
   - Energy: "Very low energy"
   - Physical: "Very unwell"
   - Focus: "Very unfocused"
3. **Expected Result**: Should trigger breathing exercise
4. **Test the breathing exercise**: Follow the guided steps, test timer, skip functionality

**Test Path 2: Low Energy User**

1. Click "Simulate Stress Notification"
2. Answer EMA questions with these responses:
   - Activity: "Working/Studying"
   - Social: "Alone"
   - Location: "Home"
   - Stress Level: "Slightly stressed"
   - Mood: "Neutral"
   - Energy: "Very low energy"
   - Physical: "Somewhat well"
   - Focus: "Very unfocused"
3. **Expected Result**: Should trigger movement exercise
4. **Test the movement exercise**: Follow the guided steps

**Test Path 3: Negative Mood User**

1. Click "Simulate Stress Notification"
2. Answer EMA questions with these responses:
   - Activity: "Relaxing"
   - Social: "Alone"
   - Location: "Home"
   - Stress Level: "Not stressed"
   - Mood: "Very negative"
   - Energy: "Moderate energy"
   - Physical: "Somewhat well"
   - Focus: "Moderately focused"
3. **Expected Result**: Should trigger meditation exercise

#### 2. Interactive Activity Testing

**Breathing Exercise Test:**

- Duration: 5 minutes total
- Steps: 5 steps with countdown timers
- Test: Timer countdown, auto-advance, manual advance, skip functionality
- Expected: "Breathe in for 4 counts" → "Hold for 4 counts" → "Breathe out for 6 counts"

**Stretching Test:**

- Duration: 3 minutes total
- Steps: 5 stretching movements
- Test: Each step has specific duration and instructions
- Expected: Overhead stretch → Shoulder rolls → Side stretch → Torso twist → Relaxation

**Meditation Test:**

- Duration: 2 minutes total
- Steps: 5 mindful steps
- Test: Eye closing, breathing awareness, present moment focus
- Expected: Close eyes → Notice breathing → Focus on present → Return to breath → Open eyes

**Movement Test:**

- Duration: 1 minute total
- Steps: 4 quick movements
- Test: Jumping jacks, breathing, shaking out, final breath
- Expected: Physical movement → Deep breath → Shake out → Another breath

#### 3. UI/UX Testing

**Navigation Testing:**

- Test all state transitions
- Test fade animations (300ms)
- Test button interactions
- Test skip functionality
- Test completion flows

**Timer Testing:**

- Verify countdown accuracy
- Test auto-advance when timer reaches 0
- Test manual advance during timer
- Test skip during timer

**Responsive Testing:**

- Test on different screen sizes
- Test button accessibility
- Test text readability
- Test color contrast

## Expected Behaviors

### ✅ Success Criteria

1. **EMA Questions**: All 8 questions display correctly with proper options
2. **Suggestion Logic**: Correct activity type triggered based on user responses
3. **Interactive Activities**:
   - Timers count down accurately
   - Steps auto-advance when timer completes
   - Manual advance works
   - Skip functionality works
   - Completion celebration displays
4. **State Management**: App returns to idle state after activity completion
5. **Error Handling**: No crashes or infinite loops

### 🐛 Common Issues to Check

1. **Timer Issues**:

   - Timer doesn't start
   - Timer doesn't count down
   - Timer doesn't auto-advance
   - Multiple timers running

2. **State Issues**:

   - App gets stuck in activity state
   - EMA answers not passed correctly
   - Suggestion logic not triggering

3. **UI Issues**:
   - Buttons not responding
   - Text not displaying
   - Animations not working
   - Layout breaking

## Debugging Tips

### Console Logging

Open browser DevTools (F12) and check console for errors:

```javascript
// Add this to debug state changes
console.log("Current state:", appState);
console.log("EMA answers:", emaAnswers);
console.log("Activity type:", activityType);
```

### State Inspection

Check React DevTools to inspect component state:

- `appState` - Current app state
- `emaAnswers` - User responses
- `activityType` - Current activity
- `activityStep` - Current step
- `activityTimer` - Timer value

### Network Testing

- Test with network disconnected
- Test with slow network
- Test API endpoints (if backend is running)

## Performance Testing

### Load Testing

- Test with multiple rapid clicks
- Test with browser tab switching
- Test with device rotation (mobile)

### Memory Testing

- Check for memory leaks
- Test long sessions
- Test multiple activity completions

## Mobile Testing

### Touch Testing

- Test touch interactions
- Test swipe gestures
- Test orientation changes
- Test keyboard interactions

### Responsive Testing

- Test on phone (375px width)
- Test on tablet (768px width)
- Test on desktop (1024px+ width)

## Automated Testing (Future)

### Unit Tests

```javascript
// Example test structure
describe("EMA App", () => {
  test("should trigger breathing exercise for high stress", () => {
    const context = {
      stress_level: "Extremely stressed",
      physical_wellbeing: "Very unwell",
    };
    const result = getSuggestionFromModel(context);
    expect(result.activityType).toBe("breathing");
  });
});
```

### Integration Tests

- Test complete user flows
- Test state transitions
- Test timer functionality
- Test activity completion

## Reporting Issues

When reporting issues, include:

1. **Steps to reproduce**
2. **Expected behavior**
3. **Actual behavior**
4. **Browser/device info**
5. **Console errors**
6. **Screenshots/videos**

## Success Metrics

- ✅ All 4 activity types work correctly
- ✅ All 8 EMA questions display properly
- ✅ Suggestion logic triggers correct activities
- ✅ Timers work accurately
- ✅ State transitions are smooth
- ✅ No crashes or errors
- ✅ Mobile responsive
- ✅ Accessible interactions

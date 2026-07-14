# gemini_utils.py
import os
import google.generativeai as genai
from typing import Dict, Any

def generate_personalized_guidance(student_text: str, emotion_data: Dict[str, Any]) -> str:
    """Generates empathetic, context-aware learning strategies using Google Gemini."""
    api_key = os.environ.get("GEMINI_API_KEY")
    
    primary = emotion_data["primary_emotion"]
    confidence = f"{emotion_data['primary_confidence']*100:.1f}%"
    secondary = emotion_data["secondary_emotion"]
    
    prompt = f"""
    You are an empathetic, world-class AI Learning Assistant. A student has submitted a query with a detected emotional state.
    
    STUDENT QUERY: "{student_text}"
    DETECTED EMOTION: {primary} (Confidence: {confidence})
    SECONDARY EMOTION: {secondary}
    
    Please provide a structured, supportive feedback response including:
    1. **Empathetic Response:** Validate their feelings naturally.
    2. **Concept Simplified:** A very short, encouraging conceptual clarification.
    3. **Immediate Action Steps:** 3 distinct, practical steps they can take right now to proceed.
    4. **Recommended Resource types:** What kinds of tools, visualizations, or materials they should find.
    
    Guidelines:
    - If Confused or Frustrated, break down problems into very small steps and use a calming tone.
    - If Bored, present a unique, high-energy angle or hands-on challenges to spark engagement.
    - If Curious, feed their interest with interesting connections or deeper theory questions.
    - Keep formatting highly readable using bullet points and clear Markdown.
    """
    
    if not api_key:
        # Fail gracefully with a rule-based fallback response block
        return get_fallback_guidance(primary, student_text)
        
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"*(Gemini API unavailable: {str(e)})*\n\n" + get_fallback_guidance(primary, student_text)

def get_fallback_guidance(emotion: str, text: str) -> str:
    """Hardcoded expert local fallback strategies to ensure zero platform downtime."""
    fallbacks = {
        "Confused": """
### 💡 Concept Simplified & Strategy
Confusion means you are right on the edge of learning something new! Let's slow down and dissect this step-by-step.

### 🛠️ Immediate Action Steps
1. **Isolate the Term:** Find the single line of code, formula, or phrase that doesn't make sense.
2. **Explain it to an Object:** Use the "rubber duck" debugging method. Talk out loud about what it is *supposed* to do.
3. **Use Hand-Traces:** Draw out loops or memory diagrams on scratch paper before looking back at the screen.
        """,
        "Frustrated": """
### ⚡ Break and Reset Strategy
Frustration blocks creative problem-solving. It's time to interrupt the cycle of trial-and-error.

### 🛠️ Immediate Action Steps
1. **Step Away for 5 Minutes:** Walk away from the screen, stretch, or grab water.
2. **Undo the Last 3 Steps:** Backtrack to when your system last worked cleanly.
3. **Write Pseudo-code:** Describe what you want to achieve in plain English without worrying about syntax constraints.
        """,
        "Curious": """
### 🚀 Exploration Strategy
Curiosity is your learning superpower! Let's channel this interest into deep mastery.

### 🛠️ Immediate Action Steps
1. **Experiment in Sandbox:** Change one parameter in your code or equation and observe what breaks.
2. **Read the Core Documentation:** Check out the source manual or specifications to see why things are built this way.
3. **Map the System:** Sketch how data flows from input to output.
        """,
        "Confident": """
### 🏆 Mastery Integration
Awesome work! Let's lock in this understanding while your momentum is high.

### 🛠️ Immediate Action Steps
1. **Optimize Your Solution:** Can you make this cleaner, faster, or more readable?
2. **Explain it to a Peer:** Teaching is the absolute best way to verify you truly understand it.
3. **Advance to the Next Tier:** Tackle a slightly harder edge case or project.
        """,
        "Bored": """
### 🎮 gamification Strategy
When material feels repetitive, change your interaction medium to wake up your attention span.

### 🛠️ Immediate Action Steps
1. **Time-box Challenges:** Set a 10-minute timer to see how much you can finish.
2. **Build a Micro-app:** Turn the dry concept into a silly script or interactive game.
3. **Change Formats:** Switch from reading to watching dynamic visual explanations.
        """
    }
    return fallbacks.get(emotion, "Let's keep breaking down this problem together. Take a breath and review your core goals.")
import base64
from io import BytesIO
from langchain_ollama import ChatOllama
import pyautogui
from langchain_core.messages import HumanMessage


# 1. Setup LLM & Specialists
llm = ChatOllama(model="qwen3.5:397b-cloud", base_url="http://localhost:11434", temperature = 0)

def capture_and_automate():
    # STEP 1: Take Screenshot
    print("📸 Taking screenshot...")
    screenshot = pyautogui.screenshot()
    
    # Convert to Base64 for the model
    buffered = BytesIO()
    screenshot.convert("RGB").save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # STEP 2: Ask the Vision Model to write automation code
    print("🧠 Analyzing screen with Qwen 3.5 Cloud...")
    message = HumanMessage(content=f"Analyze this screenshot and write automation code to perform the requested task. Screenshot: {img_str}")

    response = llm.invoke([message])
    print(f"\n--- Automation Plan ---\n{response.content}")

# Run
capture_and_automate()
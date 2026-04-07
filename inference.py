import os, time, requests
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_SPACE_URL = os.environ.get("HF_SPACE_URL", "https://shivanshd-meh.hf.space").rstrip("/")

client = InferenceClient(model="microsoft/Phi-3.5-mini-instruct", token=HF_TOKEN)

def call_llm(obs):
    prompt = f"You are a helpful agent. Answer the question based on the context.\n\nContext:\n{obs}\n\nAnswer:"
    try:
        return client.text_generation(prompt, max_new_tokens=200, temperature=0.3).strip()
    except:
        return "I don't have enough information."

def run_agent():
    print("START")
    for ep in range(1, 4):
        print(f"\n--- EPISODE {ep} ---")
        res = requests.post(f"{HF_SPACE_URL}/reset").json()
        obs = res.get("observation", "")
        print("Task:", res.get("task_type"))
        
        for step in range(1, 6):
            answer = call_llm(obs)
            print(f"Step {step} Answer: {answer[:100]}...")
            
            step_res = requests.post(f"{HF_SPACE_URL}/step", json={"action": "answer", "query": answer}).json()
            print(f"Reward: {step_res.get('reward', 0)}")
            if step_res.get("done"):
                break
            time.sleep(0.5)
    print("\nEND")

if __name__ == "__main__":
    run_agent()
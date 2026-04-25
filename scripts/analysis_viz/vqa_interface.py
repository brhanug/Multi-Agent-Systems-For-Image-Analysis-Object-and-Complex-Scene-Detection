import gradio as gr
import requests
import base64
import json
from io import BytesIO
from PIL import Image

# Endpoint of your running LLaVA-OneVision vLLM server
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_ID = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

def ask_vqa(image, question):
    # Convert image to base64 for the API
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }
        ]
    }

    response = requests.post(API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
    if response.status_code != 200:
        return f"⚠️ Error: {response.status_code}\n{response.text}"
    result = response.json()["choices"][0]["message"]["content"]
    return result

# Create Gradio UI
demo = gr.Interface(
    fn=ask_vqa,
    inputs=[
        gr.Image(type="pil", label="Upload an Image"),
        gr.Textbox(label="Ask a Question", placeholder="e.g. What is happening in this image?")
    ],
    outputs="text",
    title="🧠 LLaVA-OneVision VQA Interface",
    description="Ask visual-language questions to the model. Powered by LLaVA-OneVision Qwen2-7B (vLLM)."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
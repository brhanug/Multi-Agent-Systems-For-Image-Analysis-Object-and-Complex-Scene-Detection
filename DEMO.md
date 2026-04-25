# 🚀 Visual Historian: Project Demonstration Guide

This guide explains how to access the interactive components of the thesis project. These tools allow for real-time verification of the model agreement, taxonomy assignments, and VQA capabilities.

## 1. Available Interfaces

| Tool | Port | Purpose | Script |
| :--- | :--- | :--- | :--- |
| **VQA Interface** | 7860 | Ask natural language questions to LLaVA-OneVision. | `scripts/analysis_viz/vqa_interface.py` |
| **Archive Explorer** | 7862 | Browse the 12,110 enriched images and metadata. | `scripts/analysis_viz/archive_visualizer.py` |

## 2. Remote Access for Reviewers (Gradio Sharing)

To allow a professor or reviewer to access these tools without installing anything, use the built-in Gradio sharing feature:

1.  **Start the Interface**: Run the script as usual:
    ```bash
    python scripts/analysis_viz/vqa_interface.py
    ```
2.  **Get the Public Link**: In the terminal, look for the line:
    `Running on public URL: https://[random-id].gradio.live`
3.  **Share the URL**: Send this link to the reviewer. 
    *   **Note**: These links expire after **72 hours**. 
    *   **Tip**: For the final thesis submission, it is recommended to keep the scripts running on a server or a stable workstation during the review period.

## 3. Persistent Access (Via Ngrok)

If you need a link that lasts longer than 72 hours, use [Ngrok](https://ngrok.com/):

1.  **Install Ngrok**: `pip install pyngrok` or download the binary.
2.  **Authenticate**: `ngrok config add-authtoken [YOUR_TOKEN]`
3.  **Tunnel the Port**:
    ```bash
    ngrok http 7860  # For VQA
    ngrok http 7862  # For Archive Explorer
    ```
4.  **Copy the Link**: Provide the `https://[unique-id].ngrok-free.app` URL to the professor. This link will remain active as long as the ngrok process and the python script are running.

## 4. Prerequisite: AI Server (For VQA)

The **VQA Interface** requires the LLaVA-OneVision vLLM backend to be running:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model llava-hf/llava-onevision-qwen2-7b-ov-hf \
    --port 8000
```

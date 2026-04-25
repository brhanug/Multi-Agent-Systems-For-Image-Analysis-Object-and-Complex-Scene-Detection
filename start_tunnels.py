from pyngrok import ngrok, conf
import os
import time
import sys

# Point to the config file which includes the auth token
conf.get_default().config_path = "/data/brhanu/thesis_project/ngrok.yml"

print("🔹 Starting Ngrok tunnels...")
try:
    # Open a HTTP tunnel on the default port 80
    # <NgrokTunnel: "http://<public_sub>.ngrok.io" -> "http://localhost:80">
    vqa_tunnel = ngrok.connect(7860)
    explorer_tunnel = ngrok.connect(7862)
    api_tunnel = ngrok.connect(8001)

    print(f"VQA_URL: {vqa_tunnel.public_url}")
    print(f"EXPLORER_URL: {explorer_tunnel.public_url}")
    print(f"API_URL: {api_tunnel.public_url}")
except Exception as e:
    print(f"❌ Error starting ngrok: {e}")
    sys.exit(1)

# Keep script running to maintain tunnels
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping tunnels...")
    ngrok.kill()

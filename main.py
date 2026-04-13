import os
from dotenv import load_dotenv
from gradio_client import Client, handle_file

load_dotenv()

# 1. Connect with your Token (Crucial to avoid the login error!)
client = Client(os.getenv("HF_SPACE_ID"), token=os.getenv("Access_Token"))

# 2. Run the prediction
# Note: Since you're doing "Text-to-Video", image_input and video_input 
# can usually be set to None unless you are doing Image-to-Video.
result = client.predict(
    mode="Text-to-Video",
    prompt="A vibrant tropical fish swimming gracefully among colorful coral reefs, clear turquoise ocean, 4k, cinematic.",
    image_input=None, 
    video_input=None,
    height=384,
    width=640,
    num_frames=81,            # Start small (81 frames) to ensure it finishes quickly
    num_inference_steps=2,     # Helios-AOTI is optimized for 2-4 steps
    seed=42,
    is_amplify_first_chunk=True,
    api_name="/generate_video"
)

print(f"Video saved at: {result}")
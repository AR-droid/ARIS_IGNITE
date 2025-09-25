import time
from google.genai import Client
from google.genai import types

API_KEY = "AIzaSyCu6mJ9Yt9KmrL4QPRVso32c1UmOwWtEkg"

# Initialize client
client = Client(api_key=API_KEY)

# Prompt
prompt = "A Video explaining the paper on ML Attention is all you need with voiceover."

# Generate video
operation = client.models.generate_videos(
    model="veo-3.0-generate-001",
    prompt=prompt,
    config=types.GenerateVideosConfig(
        resolution="720p",
        duration_seconds=8,
    ),
)

print("Video generation started. Waiting for the video to be ready...")

# Wait for completion
while not operation.done:
    print("Waiting...")
    time.sleep(10)
    operation = client.operations.get(operation)

# Download the generated video
video_file = client.files.download(file=operation.response.generated_videos[0].video)

# Save directly to a file
with open("my_generated_video1.mp4", "wb") as f:
    f.write(video_file)

print("Video saved to my_generated_video1.mp4")

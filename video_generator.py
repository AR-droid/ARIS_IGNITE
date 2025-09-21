import os
import cv2
import numpy as np
import tempfile
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import base64

class VideoGenerator:
    def __init__(self, output_dir: str = "generated_videos"):
        """Initialize the video generator with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Default font (you may need to adjust the path)
        try:
            self.font = ImageFont.truetype("Arial.ttf", 24)
        except:
            self.font = ImageFont.load_default()
    
    def generate_images(self, prompts: List[Dict], image_size: Tuple[int, int] = (1280, 720)) -> List[str]:
        """
        Generate images from prompts using a text-to-image model.
        For now, this is a placeholder that creates simple colored images with text.
        In a real implementation, you would use a model like Stable Diffusion or DALL-E.
        """
        image_paths = []
        
        for i, prompt in enumerate(prompts):
            # Create a simple image with the prompt text
            img = Image.new('RGB', image_size, color=(73, 109, 137))
            d = ImageDraw.Draw(img)
            
            # Add text
            text = prompt.get('image_prompt', f'Section {i+1}')
            text_width, text_height = d.textsize(text, font=self.font)
            position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)
            d.text(position, text, fill=(255, 255, 0), font=self.font)
            
            # Save image
            img_path = self.output_dir / f"frame_{i:03d}.png"
            img.save(img_path)
            image_paths.append(str(img_path))
            
        return image_paths
    
    def generate_voiceover(self, text: str, lang: str = 'en', slow: bool = False) -> str:
        """Generate voiceover audio from text using gTTS."""
        tts = gTTS(text=text, lang=lang, slow=slow)
        audio_path = self.output_dir / "voiceover.mp3"
        tts.save(str(audio_path))
        return str(audio_path)
    
    def create_video(self, images: List[str], audio_path: str, output_path: Optional[str] = None) -> str:
        """Create a video from images and audio."""
        if not output_path:
            output_path = str(self.output_dir / "output_video.mp4")
        
        # Create video clips from images
        clips = []
        for img_path in images:
            clip = ImageClip(img_path)
            clips.append(clip)
        
        # Concatenate clips
        video = concatenate_videoclips(clips, method="compose")
        
        # Add audio
        audio = AudioFileClip(audio_path)
        video = video.set_audio(audio)
        
        # Write output
        video.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac')
        
        return output_path
    
    def generate_video_from_script(self, script: Dict, output_path: Optional[str] = None) -> Dict:
        """
        Generate a video from a script with sections containing image prompts and narration.
        
        Args:
            script: Dictionary containing video script with sections
            output_path: Optional path to save the video
            
        Returns:
            Dictionary with video path and metadata
        """
        try:
            # Generate images for each section
            image_paths = self.generate_images(script['sections'])
            
            # Generate voiceover
            full_script = script.get('script', '')
            if not full_script:
                # If no full script, concatenate section narrations
                full_script = " ".join([s.get('narration', '') for s in script['sections']])
                
            audio_path = self.generate_voiceover(full_script)
            
            # Create video
            video_path = self.create_video(image_paths, audio_path, output_path)
            
            # Clean up temporary files
            for img_path in image_paths:
                Path(img_path).unlink(missing_ok=True)
            Path(audio_path).unlink(missing_ok=True)
            
            return {
                'success': True,
                'video_path': video_path,
                'duration': sum(s.get('end_time', 5) - s.get('start_time', 0) for s in script['sections']),
                'resolution': '1280x720',
                'sections': len(script['sections'])
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Example usage
if __name__ == "__main__":
    # Example script
    example_script = {
        'title': 'Example Video',
        'script': 'This is a test video with multiple sections.',
        'sections': [
            {
                'start_time': 0,
                'end_time': 5,
                'image_prompt': 'Introduction to the topic',
                'narration': 'Welcome to this example video.'
            },
            {
                'start_time': 5,
                'end_time': 10,
                'image_prompt': 'Main content section',
                'narration': 'This is the main content of the video.'
            },
            {
                'start_time': 10,
                'end_time': 15,
                'image_prompt': 'Conclusion slide',
                'narration': 'Thank you for watching!'
            }
        ]
    }
    
    generator = VideoGenerator()
    result = generator.generate_video_from_script(example_script)
    print(f"Video generated: {result}")

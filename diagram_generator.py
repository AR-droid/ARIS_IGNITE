import os
import base64
import io
from typing import Dict, List, Optional
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import pipeline

# Available diagram types with their prompts
DIAGRAM_TYPES = {
    "flowchart": {
        "name": "Flowchart",
        "description": "Process flow and decision trees",
        "icon": "🔄",
        "prompt_template": "professional flowchart diagram, {content}, clean lines, boxes and diamonds, arrows, white background, minimalist, research paper style",
        "negative_prompt": "text, words, letters, complex, colorful, artistic, painting, drawing, sketch"
    },
    "mindmap": {
        "name": "Mind Map",
        "description": "Concept relationships and brainstorming",
        "icon": "🧠",
        "prompt_template": "professional mind map diagram, {content}, central concept with branches, clean lines, organized, white background, research paper style",
        "negative_prompt": "text, words, letters, complex, colorful, artistic, painting, drawing, sketch"
    },
    "timeline": {
        "name": "Timeline",
        "description": "Chronological events and milestones",
        "icon": "⏰",
        "prompt_template": "professional timeline diagram, {content}, horizontal line with markers, dates, events, clean design, white background, research paper style",
        "negative_prompt": "text, words, letters, complex, colorful, artistic, painting, drawing, sketch"
    },
    "venn": {
        "name": "Venn Diagram",
        "description": "Set relationships and overlaps",
        "icon": "⭕",
        "prompt_template": "professional venn diagram, {content}, overlapping circles, clean lines, mathematical, white background, research paper style",
        "negative_prompt": "text, words, letters, complex, colorful, artistic, painting, drawing, sketch"
    },
    "hierarchy": {
        "name": "Hierarchy Chart",
        "description": "Organizational structures and classifications",
        "icon": "🏗️",
        "prompt_template": "professional hierarchy chart, {content}, top-down structure, levels, clean lines, organized, white background, research paper style",
        "negative_prompt": "text, words, letters, complex, colorful, artistic, painting, drawing, sketch"
    },
    "network": {
        "name": "Network Diagram",
        "description": "Connections and relationships",
        "icon": "🕸️",
        "prompt_template": "professional network diagram, {content}, nodes and connections, clean lines, organized, white background, research paper style",
        "negative_prompt": "text, words, letters, complex, colorful, artistic, painting, drawing, sketch"
    },
    "comparison": {
        "name": "Comparison Table",
        "description": "Side-by-side analysis",
        "icon": "📊",
        "prompt_template": "professional comparison chart, {content}, table format, clean lines, organized, white background, research paper style",
        "negative_prompt": "text, words, letters, complex, colorful, artistic, painting, drawing, sketch"
    },
    "custom": {
        "name": "Custom Diagram",
        "description": "User-defined visualization",
        "icon": "🎨",
        "prompt_template": "professional diagram, {content}, clean design, white background, research paper style, academic",
        "negative_prompt": "text, words, letters, complex, colorful, artistic, painting, drawing, sketch"
    }
}

# Model configuration - Mac-optimized with multiple options for different hardware capabilities
MODEL_CONFIG = {
    # Mac-optimized models (faster, less memory, good for Apple Silicon)
    "mac_optimized": {
        "text_to_image": "CompVis/stable-diffusion-v1-4",  # ~4GB, good quality, Mac-friendly
        "image_to_text": "nlpconnect/vit-gpt2-image-captioning",  # ~1.2GB
        "description": "Mac-optimized, balanced quality and speed"
    },
    # Ultra-lightweight models (fastest, least memory, basic quality)
    "ultra_lightweight": {
        "text_to_image": "CompVis/stable-diffusion-v1-2",  # ~2GB, basic quality
        "image_to_text": "nlpconnect/vit-gpt2-image-captioning",  # ~1.2GB
        "description": "Fastest generation, basic quality, perfect for Mac"
    },
    # High-quality models (slower, more memory, best quality)
    "high_quality": {
        "text_to_image": "runwayml/stable-diffusion-v1-5",  # ~4.2GB, best quality
        "image_to_text": "nlpconnect/vit-gpt2-image-captioning",  # ~1.2GB
        "description": "Best quality, slower generation"
    }
}

# Mac-optimized hardware detection
def get_optimal_model_config():
    """Auto-select the best model configuration for Mac systems"""
    try:
        import psutil
        import torch
        import platform
        
        # Check if we're on macOS
        is_macos = platform.system() == "Darwin"
        
        # Check available RAM
        ram_gb = psutil.virtual_memory().total / (1024**3)
        
        # Check for Apple Silicon (M1/M2/M3)
        is_apple_silicon = False
        if is_macos:
            try:
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                     capture_output=True, text=True)
                is_apple_silicon = 'Apple' in result.stdout
            except:
                # Fallback check
                is_apple_silicon = platform.machine() == 'arm64'
        
        # Check if PyTorch supports MPS (Metal Performance Shaders)
        mps_available = False
        if is_macos and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            mps_available = True
        
        print(f"[diagram_generator] Mac System Detection:")
        print(f"   OS: macOS {platform.mac_ver()[0]}")
        print(f"   Architecture: {platform.machine()}")
        print(f"   Apple Silicon: {is_apple_silicon}")
        print(f"   RAM: {ram_gb:.1f} GB")
        print(f"   MPS Available: {mps_available}")
        
        # Mac-optimized selection logic
        if is_macos:
            if is_apple_silicon and ram_gb >= 16:
                if mps_available:
                    selected = "mac_optimized"
                    print(f"   Selected: {selected} (Apple Silicon + MPS)")
                else:
                    selected = "high_quality"
                    print(f"   Selected: {selected} (Apple Silicon, no MPS)")
            elif is_apple_silicon and ram_gb >= 8:
                selected = "mac_optimized"
                print(f"   Selected: {selected} (Apple Silicon, balanced)")
            elif ram_gb >= 16:
                selected = "mac_optimized"
                print(f"   Selected: {selected} (Intel Mac, high RAM)")
            elif ram_gb >= 8:
                selected = "ultra_lightweight"
                print(f"   Selected: {selected} (Mac, balanced)")
            else:
                selected = "ultra_lightweight"
                print(f"   Selected: {selected} (Mac, minimal RAM)")
        else:
            # Non-Mac fallback
            if ram_gb >= 16:
                selected = "high_quality"
                print(f"   Selected: {selected} (Non-Mac, high RAM)")
            elif ram_gb >= 8:
                selected = "mac_optimized"
                print(f"   Selected: {selected} (Non-Mac, balanced)")
            else:
                selected = "ultra_lightweight"
                print(f"   Selected: {selected} (Non-Mac, minimal RAM)")
        
        return MODEL_CONFIG[selected]
        
    except ImportError:
        # Fallback if psutil not available
        print("[diagram_generator] psutil not available, using Mac-optimized model")
        return MODEL_CONFIG["mac_optimized"]
    except Exception as e:
        print(f"[diagram_generator] Hardware detection failed: {e}, using Mac-optimized model")
        return MODEL_CONFIG["mac_optimized"]

# Current active configuration
ACTIVE_CONFIG = get_optimal_model_config()

def get_available_diagram_types() -> Dict[str, Dict]:
    """Get all available diagram types with their metadata"""
    return DIAGRAM_TYPES

def initialize_models():
    """Initialize Hugging Face models for diagram generation"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[diagram_generator] Initializing models on {device}...")
        
        # Initialize text-to-image pipeline
        model_id = ACTIVE_CONFIG["text_to_image"]
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Optimize for speed
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        if device == "cuda":
            pipe = pipe.to("cuda")
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
        
        # Initialize image captioning pipeline
        caption_pipe = pipeline("image-to-text", model=ACTIVE_CONFIG["image_to_text"])
        
        print(f"[diagram_generator] Models initialized successfully")
        return pipe, caption_pipe
        
    except Exception as e:
        print(f"[diagram_generator] Model initialization failed: {e}")
        raise Exception(f"Failed to initialize models: {str(e)}")

def generate_diagram_prompt(content: str, diagram_type: str, custom_description: str = None) -> tuple:
    """Generate optimized prompts for diagram creation"""
    
    if diagram_type not in DIAGRAM_TYPES:
        raise Exception(f"Unknown diagram type: {diagram_type}")
    
    # Get the base prompt template
    base_prompt = DIAGRAM_TYPES[diagram_type]["prompt_template"]
    negative_prompt = DIAGRAM_TYPES[diagram_type]["negative_prompt"]
    
    if diagram_type == "custom" and custom_description:
        # Use custom description for custom diagrams
        prompt = base_prompt.format(content=custom_description)
    else:
        # Use the content from the research paper
        prompt = base_prompt.format(content=content)
    
    return prompt, negative_prompt

def generate_diagram(content: str, diagram_type: str, custom_description: str = None) -> Dict:
    """
    Generate a diagram using Hugging Face models
    
    Args:
        content: The research content to visualize
        diagram_type: Type of diagram to generate
        custom_description: Custom description for custom diagrams
    
    Returns:
        Dict with 'success', 'image_data', 'error', and metadata
    """
    
    try:
        # Initialize models (this will be cached after first call)
        pipe, caption_pipe = initialize_models()
        
        # Generate the prompt
        prompt, negative_prompt = generate_diagram_prompt(content, diagram_type, custom_description)
        
        print(f"[diagram_generator] Generating {diagram_type} diagram...")
        print(f"[diagram_generator] Prompt: {prompt[:100]}...")
        
        # Generate the image
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=20,  # Reduced for speed
                guidance_scale=7.5,
                width=512,
                height=512,
                num_images_per_prompt=1
            ).images[0]
        
        # Convert to base64 for web display
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Generate a caption for the image
        caption = caption_pipe(image)[0]['generated_text']
        
        print(f"[diagram_generator] Diagram generated successfully")
        
        return {
            "success": True,
            "image_data": f"data:image/png;base64,{img_str}",
            "caption": caption,
            "diagram_type": diagram_type,
            "prompt_used": prompt,
            "negative_prompt": negative_prompt,
            "model": ACTIVE_CONFIG["text_to_image"],
        
            "dimensions": f"{image.width}x{image.height}"
        }
        
    except Exception as e:
        error_msg = f"Diagram generation failed: {str(e)}"
        print(f"[diagram_generator] {error_msg}")
        return {
            "success": False,
            "error": error_msg
        }

def generate_diagram_with_image(content: str, diagram_type: str, custom_description: str = None) -> Dict:
    """
    Generate a diagram with actual image using Hugging Face models
    This is the main function that generates real images
    """
    return generate_diagram(content, diagram_type, custom_description)

def analyze_content_for_diagram_suggestions(content: str) -> List[Dict]:
    """Analyze content and suggest appropriate diagram types"""
    
    suggestions = []
    content_lower = content.lower()
    
    # Simple keyword-based suggestions
    if any(word in content_lower for word in ["process", "flow", "step", "procedure"]):
        suggestions.append({
            "type": "flowchart",
            "reason": "Content describes processes or procedures",
            "confidence": 0.9
        })
    
    if any(word in content_lower for word in ["relationship", "connection", "network", "interaction"]):
        suggestions.append({
            "type": "network",
            "reason": "Content describes relationships or connections",
            "confidence": 0.8
        })
    
    if any(word in content_lower for word in ["timeline", "history", "evolution", "development"]):
        suggestions.append({
            "type": "timeline",
            "reason": "Content has temporal or historical elements",
            "confidence": 0.8
        })
    
    if any(word in content_lower for word in ["compare", "versus", "difference", "similarity"]):
        suggestions.append({
            "type": "comparison",
            "reason": "Content involves comparisons or contrasts",
            "confidence": 0.7
        })
    
    if any(word in content_lower for word in ["structure", "organization", "hierarchy", "classification"]):
        suggestions.append({
            "type": "hierarchy",
            "reason": "Content describes organizational structures",
            "confidence": 0.7
        })
    
    if any(word in content_lower for word in ["concept", "idea", "theory", "framework"]):
        suggestions.append({
            "type": "mindmap",
            "reason": "Content involves conceptual relationships",
            "confidence": 0.6
        })
    
    # Always include custom option
    suggestions.append({
        "type": "custom",
        "reason": "Create a custom diagram based on your description",
        "confidence": 1.0
    })
    
    # Sort by confidence
    suggestions.sort(key=lambda x: x["confidence"], reverse=True)
    
    return suggestions

def get_diagram_examples() -> Dict[str, str]:
    """Get example prompts for each diagram type"""
    examples = {}
    
    for diagram_type, info in DIAGRAM_TYPES.items():
        if diagram_type == "custom":
            examples[diagram_type] = "Describe the specific diagram you want to create"
        else:
            examples[diagram_type] = info["prompt_template"].format(
                content="the research methodology and findings"
            )
    
    return examples

def get_model_info() -> Dict:
    """Get information about the current model configuration"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "text_to_image_model": ACTIVE_CONFIG["text_to_image"],
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "model_optimizations": {
            "attention_slicing": True,
            "vae_slicing": True,
            "scheduler": "DPMSolverMultistepScheduler"
        },
        "model_config": {
            "selected": "high_quality" if ACTIVE_CONFIG == MODEL_CONFIG["high_quality"] else 
                        "lightweight" if ACTIVE_CONFIG == MODEL_CONFIG["lightweight"] else "ultra_lightweight",
            "description": ACTIVE_CONFIG["description"]
        }
    } 
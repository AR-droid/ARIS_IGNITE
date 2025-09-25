import os
import requests
import base64
import json
import threading
from datetime import datetime
from flask import copy_current_request_context
from flask import Flask, request, jsonify, render_template, send_file, Response, stream_with_context, url_for
from flask_cors import CORS
from dotenv import load_dotenv
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from google.genai import Client, types
import PyPDF2
from io import BytesIO

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Import knowledge graph
from knowledge_graph import KnowledgeGraph, EntityType, RelationType

# Initialize knowledge graph
knowledge_graph = KnowledgeGraph()

# local summarizer functions
from summarizer_optimized import generate_summaries, generate_summaries_streaming, chat_with_paper
from diagram_generator import (
    generate_diagram, 
    get_available_diagram_types, 
    analyze_content_for_diagram_suggestions,
    get_diagram_examples,
    get_model_info
)

# Import knowledge graph
from knowledge_graph import knowledge_graph

load_dotenv()

app = Flask(__name__)
# Allow all origins for all routes
CORS(app, resources={"/api/*": {"origins": "*"}})

CORE_API_KEY = os.getenv("CORE_API_KEY")
CACHE_DIR = "cached_pdfs"
os.makedirs(CACHE_DIR, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")


# 🔎 Search CORE API
@app.route("/search", methods=["GET", "POST"])
def search_papers():
    if request.method == "POST":
        data = request.get_json() or {}
        query = data.get("query", "")
    else:  # GET request
        query = request.args.get("q", "")
    if not query:
        return jsonify({"error": "Missing query"}), 400

    if not CORE_API_KEY:
        return jsonify({"error": "CORE_API_KEY not set. Please configure your API key."}), 500

    url = "https://api.core.ac.uk/v3/search/works"
    headers = {"Authorization": f"Bearer {CORE_API_KEY}"}
    payload = {"q": query, "limit": 10}

    try:
        # Increased timeout to 120 seconds for consistency with download timeout
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            return jsonify({"error": "CORE API request failed", "details": resp.text}), 500

        results = []
        for item in resp.json().get("results", []):
            paper_id = item.get("id")
            title = item.get("title", "No Title")
            authors = [a.get("name", "") for a in item.get("authors", [])]
            abstract = item.get("abstract", "No abstract available")
            
            # Get additional metadata
            year = item.get("year")
            doi = item.get("doi")
            citations = item.get("citedByCount", 0)
            download_count = item.get("downloadCount", 0)

            # ✅ Find PDF download URL
            download_url = None
            if item.get("downloadUrl"):
                download_url = item["downloadUrl"]
            elif item.get("fullText", {}).get("link"):
                download_url = item["fullText"]["link"]
            else:
                for ft in item.get("fullTextLinks", []) or []:
                    if ft.get("type") == "application/pdf" and ft.get("url"):
                        download_url = ft["url"]
                        break

            pdf_preview, local_path = None, None
            if download_url:
                local_path = os.path.join(CACHE_DIR, f"{paper_id}.pdf")
                
                # Start background download if PDF doesn't exist
                if not os.path.exists(local_path):
                    @copy_current_request_context
                    def download_pdf(pdf_url, path):
                        try:
                            r = requests.get(pdf_url, timeout=120)
                            ct = r.headers.get("content-type", "").lower()
                            if r.status_code == 200 and ("pdf" in ct or pdf_url.lower().endswith(".pdf")):
                                with open(path, "wb") as f:
                                    f.write(r.content)
                                print(f"[background download] Successfully downloaded {pdf_url}")
                        except Exception as e:
                            print(f"[background download error] {pdf_url} -> {e}")
                    
                    # Start background download
                    threading.Thread(
                        target=download_pdf,
                        args=(download_url, local_path),
                        daemon=True
                    ).start()
                else:
                    # If PDF exists, generate preview
                    try:
                        with open(local_path, "rb") as f:
                            encoded = base64.b64encode(f.read()).decode("utf-8")
                            pdf_preview = f"data:application/pdf;base64,{encoded}"
                    except Exception as e:
                        print(f"[encode error] {local_path} -> {e}")

            # Add result immediately with placeholder for PDF preview
            results.append({
                "id": paper_id,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "year": year,
                "doi": doi,
                "citations": citations,
                "download_count": download_count,
                "pdf_preview": pdf_preview,  # Will be None initially if PDF needs downloading
                "download_url": f"/download/{paper_id}" if os.path.exists(os.path.join(CACHE_DIR, f"{paper_id}.pdf")) else None,
                "pdf_status": "available" if os.path.exists(os.path.join(CACHE_DIR, f"{paper_id}.pdf")) else "pending"
            })

        return jsonify({"papers": results, "total": len(results), "query": query})
        
    except requests.exceptions.Timeout:
        return jsonify({"error": "Search request timed out. Please try again."}), 408
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Network error occurred", "details": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "Unexpected error occurred", "details": str(e)}), 500


# 📥 Download cached PDF
@app.route("/download/<paper_id>")
def download_paper(paper_id):
    path = os.path.join(CACHE_DIR, f"{paper_id}.pdf")
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, mimetype="application/pdf")


# 📝 Summarize with RAG (Legacy - waits for all summaries)
@app.route('/summarize/<paper_id>', methods=['GET'])
def summarize_paper(paper_id):
    """Legacy endpoint that waits for all summaries to be generated."""
    try:
        pdf_path = os.path.join(CACHE_DIR, f"{paper_id}.pdf")
        if not os.path.exists(pdf_path):
            return jsonify({"error": "Paper not found in cache"}), 404
            
        # Use the optimized summarizer
        summaries = generate_summaries(pdf_path, paper_id)
        return jsonify({"summaries": summaries})
        
    except Exception as e:
        error_msg = f"Error generating summaries: {str(e)}"
        print(f"[summarize_paper] {error_msg}")
        return jsonify({"error": error_msg}), 500


# 📝 Summarize with RAG (Streaming - shows summaries as they're ready)
@app.route('/summarize-stream/<paper_id>', methods=['GET'])
def summarize_paper_streaming(paper_id):
    """Streaming endpoint that yields summaries as they're generated."""
    try:
        pdf_path = os.path.join(CACHE_DIR, f"{paper_id}.pdf")
        if not os.path.exists(pdf_path):
            return jsonify({"error": "Paper not found in cache"}), 404
            
        def generate():
            try:
                for level, summary, status in generate_summaries_streaming(pdf_path, paper_id):
                    # Skip empty or error summaries
                    if status == "error" and not summary:
                        summary = f"Error generating {level} summary"
                    
                    # Send the event
                    event = {'level': level, 'summary': summary, 'status': status}
                    yield f"data: {json.dumps(event)}\n\n"
                    
                    # Add a small delay to ensure the client can process the event
                    import time
                    time.sleep(0.05)  # Reduced delay for better responsiveness
                    
            except Exception as e:
                error_msg = f"Error in summary generation: {str(e)}"
                print(f"[summarize_paper_streaming] {error_msg}")
                yield f"data: {json.dumps({'level': 'error', 'summary': error_msg, 'status': 'error'})}\n\n"
                
        return Response(stream_with_context(generate()), mimetype='text/event-stream')
        
    except Exception as e:
        error_msg = f"Failed to start summary generation: {str(e)}"
        print(f"[summarize_paper_streaming] {error_msg}")
        return jsonify({"error": error_msg}), 500


# 💬 Chat with RAG
@app.route('/chat/<paper_id>', methods=['POST'])
def chat_paper(paper_id):
    """Chat with the paper using RAG."""
    try:
        data = request.get_json() or {}
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
            
        pdf_path = os.path.join(CACHE_DIR, f"{paper_id}.pdf")
        if not os.path.exists(pdf_path):
            return jsonify({"error": "Paper not found in cache"}), 404
            
        # Use ThreadPoolExecutor with timeout for thread-safe operation
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as ThreadTimeoutError
        import functools
        
        # Create a wrapper function with the required arguments
        def process_chat():
            return chat_with_paper(pdf_path, paper_id, query)
        
        # Use ThreadPoolExecutor to run the chat with a timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(process_chat)
            try:
                # Set timeout to 60 seconds
                response = future.result(timeout=60)
                return jsonify({"response": response})
                
            except ThreadTimeoutError:
                return jsonify({
                    "error": "Chat operation timed out. Please try again with a more specific query."
                }), 408
                
            except Exception as e:
                error_msg = f"Error in chat: {str(e)}"
                print(f"[chat_paper] {error_msg}")
                return jsonify({"error": error_msg}), 500
        
    except Exception as e:
        error_msg = f"Failed to process chat request: {str(e)}"
        print(f"[chat_paper] {error_msg}")
        return jsonify({"error": error_msg}), 500


# 🎨 Diagram Generation Endpoints
@app.route("/diagram-types")
def get_diagram_types():
    """Get all available diagram types"""
    try:
        types = get_available_diagram_types()
        return jsonify({"diagram_types": types})
    except Exception as e:
        return jsonify({"error": "Failed to get diagram types", "details": str(e)}), 500


@app.route("/diagram-suggestions/<paper_id>")
def get_diagram_suggestions(paper_id):
    """Get diagram type suggestions based on paper content"""
    path = os.path.join(CACHE_DIR, f"{paper_id}.pdf")
    if not os.path.exists(path):
        return jsonify({"error": "Paper not found"}), 404
    
    try:
        from summarizer import extract_text_from_pdf
        text = extract_text_from_pdf(path)
        suggestions = analyze_content_for_diagram_suggestions(text)
        return jsonify({"suggestions": suggestions})
    except Exception as e:
        return jsonify({"error": "Failed to analyze content", "details": str(e)}), 500


@app.route("/generate-diagram/<paper_id>", methods=["POST"])
def create_diagram(paper_id):
    """Generate a diagram based on paper content"""
    data = request.get_json() or {}
    diagram_type = data.get("diagram_type", "custom")
    custom_description = data.get("custom_description", "")
    
    if not diagram_type:
        return jsonify({"error": "Diagram type is required"}), 400
    
    path = os.path.join(CACHE_DIR, f"{paper_id}.pdf")
    if not os.path.exists(path):
        return jsonify({"error": "Paper not found"}), 404
    
    try:
        from summarizer import extract_text_from_pdf
        text = extract_text_from_pdf(path)
        
        # Generate the diagram
        result = generate_diagram(text, diagram_type, custom_description)
        
        if result["success"]:
            return jsonify(result)
        else:
            return jsonify({"error": result["error"]}), 500
            
    except Exception as e:
        return jsonify({"error": "Diagram generation failed", "details": str(e)}), 500


@app.route("/diagram-examples")
def get_diagram_examples_endpoint():
    """Get example prompts for diagram types"""
    try:
        examples = get_diagram_examples()
        return jsonify({"examples": examples})
    except Exception as e:
        return jsonify({"error": "Failed to get examples", "details": str(e)}), 500


@app.route('/diagram/model-info', methods=['GET'])
def get_model_info_endpoint():
    """Get information about the Hugging Face models"""
    info = get_model_info()
    return jsonify(info)


@app.route('/generate-video/<paper_id>', methods=['POST'])
def generate_video(paper_id):
    """
    Generate a video summary of the paper using Gemini Veo 3 API with prompts from Groq's Llama 3.3 70B.
    
    This endpoint will:
    1. Get the paper content
    2. Use Groq's Llama 3.3 70B to generate a detailed video script and prompts
    3. Use Gemini Veo 3 to generate the video
    4. Return the video URL
    """
    try:
        # Get the paper content
        paper_info = get_paper_info(paper_id)
        if not paper_info or 'error' in paper_info:
            return jsonify({"error": "Failed to get paper info"}), 404
            
        # Get the paper text
        paper_text = paper_info.get('abstract', '') + " " + paper_info.get('full_text', '')
        
        # Initialize Groq client for Llama 3.3 70B
        from langchain_groq import ChatGroq
        import os
        import json
        from google.genai import Client, types
        from pathlib import Path
        import time
        
        # Initialize Groq client
        groq_api_key = os.getenv("GROK")
        if not groq_api_key:
            return jsonify({"error": "Groq API key not found"}), 500
            
        llm = ChatGroq(
            temperature=0.3,
            model_name="llama-3.3-70b-versatile",
            groq_api_key=groq_api_key
        )
        
        # Generate a focused video prompt using the paper's content
        prompt = f"""
        Create a concise video prompt (1-2 sentences) based on this research paper's content.
        Focus on the most important findings and key points from the paper along with voiceover.
        
        Paper title: {paper_info.get('title', 'Research Paper')}
        
        Paper content (first 8000 characters):
        {paper_text[:8000]}
        
        Guidelines:
        - Be specific about the paper's actual content
        - Mention key terms, concepts, or findings from the paper
        - Keep it focused and educational
        - Avoid generic phrases unless they specifically relate to the paper
        
        Example format:
        "A video explaining [specific finding/concept] from the paper, 
        showing [key visual elements] that illustrate [main points]. With a brief voiceover"
        """
        
        # Call the LLM
        response = llm.invoke(prompt)
        
        try:
            # Get the prompt text directly
            video_prompt = response.content.strip('"\'').strip()
            
            # Initialize Gemini client
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key:
                return jsonify({"error": "Gemini API key not found"}), 500
                
            client = Client(api_key=gemini_api_key)
            
            # Create video directory if it doesn't exist
            video_dir = Path("static/videos")
            video_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate video using Gemini Veo 3 with fixed parameters
            try:
                print(f"Sending video generation request with prompt: {video_prompt}")
                operation = client.models.generate_videos(
                    model="veo-3.0-generate-001",
                    prompt=video_prompt,
                    config=types.GenerateVideosConfig(
                        resolution="1080p",
                        duration_seconds=8,  # 8 seconds for testing, can be increased later
                        aspect_ratio="16:9"
                    ),
                )
            except Exception as e:
                print(f"Error in video generation request: {str(e)}")
                return jsonify({
                    "error": f"Failed to start video generation: {str(e)}",
                    "prompt_used": video_prompt
                }), 500
            
            print("Video generation started. Waiting for the video to be ready...")
            print(f"Using prompt: {video_prompt}")
            
            # Wait for completion with progress updates
            start_time = time.time()
            timeout = 600  # 10 minute timeout
            
            while not operation.done:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    return jsonify({
                        "error": "Video generation timed out",
                        "status": "timeout"
                    }), 504
                
                print(f"Waiting for video generation... ({elapsed:.0f}s elapsed)")
                time.sleep(10)
                operation = client.operations.get(operation)
            
            # Check if generation was successful
            if not hasattr(operation, 'response') or not hasattr(operation.response, 'generated_videos') or not operation.response.generated_videos:
                return jsonify({
                    "error": "Video generation failed: No video in response",
                    "operation_status": str(operation)
                }), 500
            
            # Download the generated video
            try:
                video_file = client.files.download(file=operation.response.generated_videos[0].video)
                
                # Save the video with paper ID and timestamp
                timestamp = int(time.time())
                video_filename = f"{paper_id}_{timestamp}.mp4"
                video_path = video_dir / video_filename
                
                with open(video_path, "wb") as f:
                    f.write(video_file)
                
                # Generate thumbnail
                thumbnail_filename = f"{paper_id}_{timestamp}_thumb.jpg"
                thumbnail_path = video_dir / thumbnail_filename
                
                try:
                    import cv2
                    import numpy as np
                    
                    # Read the video file
                    cap = cv2.VideoCapture(str(video_path))
                    ret, frame = cap.read()
                    if ret:
                        # Convert BGR to RGB and resize
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, (320, 180))
                        # Save as JPEG with 90% quality
                        cv2.imwrite(str(thumbnail_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                        print(f"Thumbnail generated: {thumbnail_path}")
                    else:
                        print("Failed to capture video frame for thumbnail")
                        thumbnail_path = None
                    cap.release()
                except Exception as e:
                    print(f"Error generating thumbnail: {e}")
                    thumbnail_path = None
                
                # Prepare response with metadata
                response_data = {
                    "status": "completed",
                    "video_url": f"/static/videos/{video_filename}",
                    "thumbnail_url": f"/static/videos/{thumbnail_filename}" if thumbnail_path else None,
                    "metadata": {
                        "title": f"Research Summary: {paper_info.get('title', 'Paper')}",
                        "description": f"AI-generated summary of research paper: {paper_info.get('title', '')}",
                        "duration": 8,
                        "resolution": "1920x1080",
                        "aspect_ratio": "16:9",
                        "generated_at": datetime.now().isoformat(),
                        "paper_source": paper_id
                    },
                    "prompt_used": video_prompt,
                    "message": "Video generated successfully with Gemini Veo 3"
                }
                
                print(f"Video generation completed in {time.time() - start_time:.2f} seconds")
                return jsonify(response_data)
                
            except Exception as e:
                print(f"Error processing video file: {e}")
                return jsonify({
                    "error": f"Failed to process video file: {str(e)}",
                    "prompt_used": video_prompt
                }), 500
            
        except json.JSONDecodeError as e:
            return jsonify({
                "error": "Failed to parse video script from LLM response",
                "llm_response": response.content,
                "exception": str(e)
            }), 500
        except Exception as e:
            import traceback
            return jsonify({
                "error": f"Video generation failed: {str(e)}",
                "traceback": traceback.format_exc()
            }), 500
            
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Failed to generate video: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500


# 🔍 Get paper info
@app.route("/paper/<paper_id>")
def get_paper_info(paper_id):
    """Get paper information including metadata and content."""
    path = os.path.join(CACHE_DIR, f"{paper_id}.pdf")
    if not os.path.exists(path):
        return {"error": "Paper not found"}, 404
    
    try:
        # Get file metadata
        file_size = os.path.getsize(path)
        file_size_mb = round(file_size / (1024 * 1024), 2)
        
        # Extract text from PDF
        full_text = extract_text_from_pdf(path)
        if not full_text:
            return {"error": "Failed to extract text from PDF"}, 500
        
        # Extract first 1000 characters as a preview
        preview = full_text[:1000] + (full_text[1000:] and '...')
        
        # Try to extract title from first few lines
        title = f"Research Paper {paper_id}"
        first_lines = [line.strip() for line in full_text.split('\n') if line.strip()]
        if first_lines:
            title = first_lines[0][:200]  # Use first non-empty line as title, max 200 chars
        
        return {
            "id": paper_id,
            "title": title,
            "abstract": preview,  # Using first 1000 chars as abstract
            "full_text": full_text,
            "year": datetime.now().year,
            "file_size_mb": file_size_mb,
            "cached": True
        }
    except Exception as e:
        return jsonify({"error": "Failed to get paper info", "details": str(e)}), 500


# 🧹 Clear cache
@app.route("/clear-cache", methods=["POST"])
def clear_cache():
    try:
        import shutil
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            os.makedirs(CACHE_DIR, exist_ok=True)
        return jsonify({"message": "Cache cleared successfully"})
    except Exception as e:
        return jsonify({"error": "Failed to clear cache", "details": str(e)}), 500


# 📊 Health check
@app.route("/health")
def health_check():
    try:
        # Check if CORE API key is configured
        api_key_status = "configured" if CORE_API_KEY else "missing"
        
        # Check cache directory
        cache_status = "accessible" if os.access(CACHE_DIR, os.W_OK) else "not accessible"
        
        # Check if any PDFs are cached
        cached_pdfs = len([f for f in os.listdir(CACHE_DIR) if f.endswith('.pdf')])
        
        return jsonify({
            "status": "healthy",
            "core_api_key": api_key_status,
            "cache_directory": cache_status,
            "cached_pdfs": cached_pdfs,
            "timestamp": str(datetime.now())
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


# Knowledge Graph Endpoints
@app.route("/api/knowledge-graph/paper/<paper_id>", methods=["GET"])
def get_knowledge_graph_for_paper(paper_id: str):
    """Get knowledge graph data for a specific paper."""
    try:
        # Try to get paper info first
        paper_info = get_paper_info(paper_id)
        if not paper_info:
            return jsonify({"error": "Paper not found"}), 404
        
        # Add paper to knowledge graph if not already present
        if paper_id not in knowledge_graph.graph:
            knowledge_graph.add_paper(
                paper_id=paper_id,
                title=paper_info.get("title", ""),
                abstract=paper_info.get("abstract", ""),
                authors=paper_info.get("authors", []),
                metadata={
                    "year": paper_info.get("year"),
                    "citations": paper_info.get("citations", 0),
                    "downloads": paper_info.get("downloads", 0)
                }
            )
        
        # Get the complete graph data
        graph_data = knowledge_graph.get_graph_data()
        
        # Add the current paper to the graph data if it's not already there
        paper_node = {
            "id": paper_id,
            "type": "PAPER",
            "title": paper_info.get("title", ""),
            "abstract": paper_info.get("abstract", ""),
            "year": paper_info.get("year"),
            "citations": paper_info.get("citations", 0),
            "downloads": paper_info.get("downloads", 0),
            "label": paper_info.get("title", "")[:50] + ("..." if len(paper_info.get("title", "")) > 50 else ""),
            "size": 15,
            "group": "paper"
        }
        
        # Add paper node if not already in nodes
        if not any(node["id"] == paper_id for node in graph_data["nodes"]):
            graph_data["nodes"].append(paper_node)
            
            # Add author relationships
            for author in paper_info.get("authors", []):
                author_id = f"author_{author.get('id', author.get('name', '').lower().replace(' ', '_'))}"
                
                # Add author node if not exists
                if not any(node["id"] == author_id for node in graph_data["nodes"]):
                    graph_data["nodes"].append({
                        "id": author_id,
                        "type": "AUTHOR",
                        "name": author.get("name", ""),
                        "label": author.get("name", ""),
                        "size": 10,
                        "group": "author"
                    })
                
                # Add authored_by relationship
                graph_data["links"].append({
                    "source": paper_id,
                    "target": author_id,
                    "type": "AUTHORED_BY",
                    "value": 1
                })
        
        return jsonify(graph_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/knowledge-graph/concept/<concept>")
def get_concept_network(concept: str):
    """Get network for a specific concept."""
    try:
        network = knowledge_graph.get_concept_network(concept)
        return jsonify(network)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/knowledge-graph/author/<author_name>", methods=["GET"])
def get_author_network(author_name: str):
    """Get network for a specific author."""
    try:
        author_id = knowledge_graph._get_author_id({"name": author_name})
        network = knowledge_graph.get_author_network(author_id)
        return jsonify(network)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/knowledge-graph/demo", methods=["GET"])
def get_demo_knowledge_graph():
    """Get a demo knowledge graph with sample research concepts."""
    try:
        # Clear existing graph for demo
        knowledge_graph.graph = nx.DiGraph()
        
        # Add research paper
        paper_id = "paper_123"
        knowledge_graph.graph.add_node(
            paper_id,
            type=EntityType.PAPER.value,
            label="A Survey on Large Language Models",
            abstract="A comprehensive survey on the latest advancements in large language models.",
            year=2023
        )
        
        # Add authors
        authors = ["Alice Johnson", "Bob Smith", "Carol Williams"]
        for i, author_name in enumerate(authors):
            author_id = f"author_{i+1}"
            knowledge_graph.graph.add_node(
                author_id,
                type=EntityType.AUTHOR.value,
                label=author_name,
                name=author_name
            )
            # Link authors to paper
            knowledge_graph.graph.add_edge(
                paper_id,
                author_id,
                type=RelationType.AUTHORED_BY.value,
                label="authored by"
            )
        
        # Add research concepts (15 keywords)
        concepts = [
            ("LLM", "Large Language Models"),
            ("NLP", "Natural Language Processing"),
            ("Transformers", "Transformer Architecture"),
            ("Attention", "Attention Mechanism"),
            ("Fine-tuning", "Model Fine-tuning"),
            ("Prompting", "Prompt Engineering"),
            ("RLHF", "Reinforcement Learning from Human Feedback"),
            ("Multimodal", "Multimodal Learning"),
            ("Bias", "Algorithmic Bias"),
            ("Ethics", "AI Ethics"),
            ("Evaluation", "Model Evaluation"),
            ("Retrieval", "Retrieval-Augmented Generation"),
            ("Efficiency", "Model Efficiency"),
            ("Scaling", "Model Scaling"),
            ("Applications", "Real-world Applications")
        ]
        
        # Add concepts and link to paper
        for i, (concept_id, concept_label) in enumerate(concepts):
            knowledge_graph.graph.add_node(
                concept_id,
                type=EntityType.CONCEPT.value,
                label=concept_label,
                name=concept_label
            )
            # Link concepts to paper
            knowledge_graph.graph.add_edge(
                paper_id,
                concept_id,
                type=RelationType.ADDRESSES.value,
                label="addresses"
            )
            
            # Add some relationships between concepts
            if i > 0:
                knowledge_graph.graph.add_edge(
                    concepts[i-1][0],
                    concept_id,
                    type=RelationType.RELATED_TO.value if hasattr(RelationType, 'RELATED_TO') else "RELATED_TO",
                    label="related to"
                )
        
        # Add some additional relationships
        knowledge_graph.graph.add_edge("LLM", "Transformers", type="USES", label="uses")
        knowledge_graph.graph.add_edge("Transformers", "Attention", type="UTILIZES", label="utilizes")
        knowledge_graph.graph.add_edge("LLM", "Fine-tuning", type="REQUIRES", label="requires")
        knowledge_graph.graph.add_edge("LLM", "Prompting", type="SUPPORTS", label="supports")
        knowledge_graph.graph.add_edge("RLHF", "Ethics", type="ADDRESSES", label="addresses")
        knowledge_graph.graph.add_edge("Bias", "Ethics", type="RELATES_TO", label="relates to")
        knowledge_graph.graph.add_edge("Evaluation", "Applications", type="INFORMS", label="informs")
        
        # Return the graph data
        graph_data = knowledge_graph.get_graph_data()
        return jsonify({
            "status": "success",
            "message": "Demo knowledge graph generated with 15 research concepts",
            "graph": graph_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Initialize spaCy model for knowledge graph
_nlp_initialized = False

def initialize_nlp():
    global _nlp_initialized
    if _nlp_initialized:
        return
        
    try:
        import spacy
        # Download the English language model if not already present
        try:
            spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        _nlp_initialized = True
    except Exception as e:
        app.logger.error(f"Failed to initialize spaCy: {str(e)}")
        raise

# Register the initialization to run before each request
@app.before_request
def before_request():
    if not _nlp_initialized:
        initialize_nlp()


@app.route('/3dvis')
def molecule_viewer():
    """Serve the 3D molecule viewer interface."""
    # Get the title from the query parameters, default to 'Molecule' if not provided
    title = request.args.get('title', 'Molecule')
    return render_template('3dvis.html', molecule_title=title)

@app.route('/api/seo/analyze', methods=['POST'])
def analyze_seo():
    """Analyze text for SEO, AI content, and plagiarism."""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Basic text analysis
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
        
        # Calculate basic metrics
        word_count = len(words)
        sentence_count = len(sent_tokenize(text))
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Check for common AI patterns (simplified example)
        ai_patterns = [
            (r'\b(as an ai|as a language model|i am an ai|i am a language model)\b', 'ai_self_reference'),
            (r'\b(however, it is important to note|it is worth noting|it is important to remember)\b', 'ai_hedging'),
            (r'\b(in conclusion|to summarize|in summary|ultimately|overall)\b', 'ai_conclusion_phrases')
        ]
        
        ai_detection = {}
        for pattern, label in ai_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                ai_detection[label] = len(matches)
        
        # Calculate AI probability (simplified example)
        ai_probability = min(100, sum(ai_detection.values()) * 10)
        
        # Generate readability score (Flesch-Kincaid Reading Ease)
        def calculate_readability(text):
            sentences = sent_tokenize(text)
            words = word_tokenize(text.lower())
            syllables = sum([sum(1 for letter in word if letter in 'aeiouy') for word in words])
            
            if len(sentences) == 0 or len(words) == 0:
                return 0
                
            # Flesch-Kincaid Reading Ease
            try:
                score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllables / len(words))
                return max(0, min(100, score))
            except ZeroDivisionError:
                return 0
        
        readability_score = calculate_readability(text)
        
        # Generate keyword density (top 10)
        word_freq = {}
        for word in filtered_words:
            if len(word) > 2:  # Only consider words longer than 2 characters
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and get top 10
        keyword_density = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Calculate keyword density percentages
        total_words = len(filtered_words)
        keyword_density = {k: round((v / total_words) * 100, 2) for k, v in keyword_density.items()}
        
        # Check for passive voice (simplified example)
        passive_voice_matches = len(re.findall(r'\b(am|are|were|being|is|been|was|be)\b\s+\w+ed\b', text, re.IGNORECASE))
        
        return jsonify({
            "metrics": {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_word_length": round(avg_word_length, 2),
                "avg_sentence_length": round(avg_sentence_length, 2),
                "readability_score": round(readability_score, 2),
                "passive_voice_instances": passive_voice_matches,
                "ai_probability": ai_probability,
                "keyword_density": keyword_density
            },
            "ai_detection": ai_detection,
            "seo_suggestions": [
                "Use more headings and subheadings" if len(re.findall(r'<h[1-6]', text)) < 3 else "Good heading structure",
                "Consider adding more images with alt text" if len(re.findall(r'<img[^>]*alt="[^"]*"', text)) < 1 else "Good image alt text usage",
                "Add internal links to related content" if len(re.findall(r'href="/[^"]*"', text)) < 2 else "Good internal linking"
            ]
        })
        
    except Exception as e:
        return jsonify({
            "error": "Failed to analyze content",
            "details": str(e)
        }), 500

@app.route('/api/citations', methods=['POST'])
def generate_citation():
    """
    Generate citations in various formats with dummy content.
    
    Request JSON format:
    {
        "title": "Sample Research Paper",
        "authors": ["John Doe", "Jane Smith"],
        "year": "2023",
        "journal": "Journal of Sample Research",
        "volume": "12",
        "issue": "3",
        "pages": "45-67",
        "doi": "10.1234/sample.2023.001",
        "url": "https://example.com/paper123"
    }
    
    Response format:
    {
        "apa": "Doe, J., & Smith, J. (2023). Sample Research Paper. Journal of Sample Research, 12(3), 45-67. https://doi.org/10.1234/sample.2023.001",
        "mla": "Doe, John, and Jane Smith. \"Sample Research Paper.\" Journal of Sample Research, vol. 12, no. 3, 2023, pp. 45-67, https://doi.org/10.1234/sample.2023.001.",
        "chicago": "Doe, John, and Jane Smith. 2023. \"Sample Research Paper.\" Journal of Sample Research 12 (3): 45-67. https://doi.org/10.1234/sample.2023.001.",
        "ieee": "J. Doe and J. Smith, \"Sample Research Paper,\" Journal of Sample Research, vol. 12, no. 3, pp. 45-67, 2023, doi: 10.1234/sample.2023.001.",
        "harvard": "Doe, J. and Smith, J. (2023) 'Sample Research Paper', Journal of Sample Research, 12(3), pp. 45-67. doi: 10.1234/sample.2023.001.",
        "bibtex": "@article{doe2023sample,\n  title={Sample Research Paper},\n  author={Doe, John and Smith, Jane},\n  journal={Journal of Sample Research},\n  volume={12},\n  number={3},\n  pages={45--67},\n  year={2023},\n  publisher={Sample Publisher},\n  doi={10.1234/sample.2023.001},\n  url={https://example.com/paper123}\n}"
    }
    """
    try:
        data = request.get_json()
        
        # Dummy data for demonstration
        citation_data = {
            "title": data.get('title', 'Sample Research Paper'),
            "authors": data.get('authors', ["John Doe", "Jane Smith"]),
            "year": data.get('year', '2023'),
            "journal": data.get('journal', 'Journal of Sample Research'),
            "volume": data.get('volume', '12'),
            "issue": data.get('issue', '3'),
            "pages": data.get('pages', '45-67'),
            "doi": data.get('doi', '10.1234/sample.2023.001'),
            "url": data.get('url', 'https://example.com/paper123')
        }
        
        # Format authors for different citation styles
        authors = citation_data['authors']
        if len(authors) == 1:
            apa_authors = f"{authors[0].split()[-1]}"
            mla_authors = f"{authors[0]}"
        else:
            apa_authors = ", ".join([f"{name.split()[-1]}" for name in authors[:-1]])
            apa_authors += f" & {authors[-1].split()[-1]}" if len(authors) > 1 else ""
            mla_authors = ", ".join([f"{name}" for name in authors[:-1]])
            mla_authors += f", and {authors[-1]}" if len(authors) > 1 else ""
        
        # Generate citations in different formats
        doi_key = citation_data['doi'].replace('.', '')
        title = citation_data['title']
        journal = citation_data['journal']
        volume = citation_data['volume']
        issue = citation_data['issue']
        pages = citation_data['pages']
        year = citation_data['year']
        doi = citation_data['doi']
        url = citation_data['url']
        
        # Format author strings
        author_list = '" and "'.join(authors) if len(authors) > 1 else authors[0]
        
        citations = {
            "apa": (
                f"{apa_authors} ({year}). {title}. {journal}, "
                f"{volume}({issue}), {pages}. https://doi.org/{doi}"
            ),
            "mla": (
                f"{mla_authors}. \"{title}.\" {journal}, vol. {volume}, "
                f"no. {issue}, {year}, pp. {pages}, https://doi.org/{doi}."
            ),
            "chicago": (
                f"{', '.join(authors)}. {year}. \"{title}.\" {journal} "
                f"{volume}, no. {issue}: {pages}. https://doi.org/{doi}."
            ),
            "ieee": (
                f"{', '.join([f'{name[0]}. {name.split()[-1]}' for name in authors])}, "
                f'"{title}," {journal}, vol. {volume}, no. {issue}, pp. {pages}, '
                f"{year}, doi: {doi}."
            ),
            "harvard": (
                f"{', '.join([f'{name.split()[-1]}, {name[0]}' for name in authors])} "
                f"({year}) '{title}', {journal}, {volume}({issue}), "
                f"pp. {pages}. doi: {doi}."
            ),
            "bibtex": (
                f"@article{{{doi_key},\n"
                f"  title={{{title}}},\n"
                f'  author={{"{author_list}"}},\n'
                f"  journal={{{journal}}},\n"
                f"  volume={{{volume}}},\n"
                f"  number={{{issue}}},\n"
                f"  pages={{{pages}}},\n"
                f"  year={{{year}}},\n"
                f"  doi={{{doi}}},\n"
                f"  url={{{url}}}\n"
                "}"
            )
        }
        
        return jsonify(citations)
        
    except Exception as e:
        return jsonify({
            "error": "Failed to generate citations",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001, host='0.0.0.0')
import os
import requests
import base64
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv

# Import knowledge graph
from knowledge_graph import KnowledgeGraph

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
CORS(app)

CORE_API_KEY = os.getenv("CORE_API_KEY")
CACHE_DIR = "cached_pdfs"
os.makedirs(CACHE_DIR, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")


# 🔎 Search CORE API
@app.route("/search", methods=["POST"])
def search_papers():
    data = request.get_json() or {}
    query = data.get("query", "")
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
                if not os.path.exists(local_path):
                    try:
                        # Increased timeout to 120 seconds to handle slower downloads
                        r = requests.get(download_url, timeout=120)
                        ct = r.headers.get("content-type", "").lower()
                        if r.status_code == 200 and ("pdf" in ct or download_url.lower().endswith(".pdf")):
                            with open(local_path, "wb") as f:
                                f.write(r.content)
                    except Exception as e:
                        print(f"[download error] {download_url} -> {e}")

                if os.path.exists(local_path):
                    try:
                        with open(local_path, "rb") as f:
                            encoded = base64.b64encode(f.read()).decode("utf-8")
                            pdf_preview = f"data:application/pdf;base64,{encoded}"
                    except Exception as e:
                        print(f"[encode error] {local_path} -> {e}")

            results.append({
                "id": paper_id,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "year": year,
                "doi": doi,
                "citations": citations,
                "download_count": download_count,
                "pdf_preview": pdf_preview,
                "download_url": f"/download/{paper_id}" if local_path else None
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
            
        # Add a timeout to prevent hanging requests
        import signal
        from functools import wraps
        
        class TimeoutError(Exception):
            pass
            
        def timeout_handler(signum, frame):
            raise TimeoutError("Chat operation timed out")
            
        # Set the timeout to 60 seconds
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)
        
        try:
            response = chat_with_paper(pdf_path, paper_id, query)
            signal.alarm(0)  # Disable the alarm
            return jsonify({"response": response})
            
        except TimeoutError:
            return jsonify({"error": "Chat operation timed out. Please try again with a more specific query."}), 408
            
        except Exception as e:
            error_msg = f"Error in chat: {str(e)}"
            print(f"[chat_paper] {error_msg}")
            return jsonify({"error": error_msg}), 500
            
        finally:
            signal.alarm(0)  # Ensure the alarm is always disabled
        
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
    Generate a video summary of the paper using Groq LLM and OpenCV.
    
    This endpoint will:
    1. Get the paper content
    2. Use Groq LLM to generate a script and image prompts
    3. Generate images using the prompts
    4. Create a video with voiceover using TTS
    5. Return the video URL
    """
    try:
        # Get the paper content
        paper_info = get_paper_info(paper_id)
        if not paper_info or 'error' in paper_info:
            return jsonify({"error": "Failed to get paper info"}), 404
            
        # Get the paper text (you might need to adjust this based on your data structure)
        paper_text = paper_info.get('abstract', '') + " " + paper_info.get('full_text', '')
        
        # Initialize Groq client
        from langchain_groq import ChatGroq
        import os
        from video_generator import VideoGenerator
        from pathlib import Path
        import json
        
        groq_api_key = os.getenv("GROK")
        if not groq_api_key:
            return jsonify({"error": "Groq API key not found"}), 500
            
        llm = ChatGroq(
            temperature=0.3,
            model_name="llama-3-70b-8192",
            groq_api_key=groq_api_key
        )
        
        # Generate video script and image prompts using LLM
        prompt = f"""
        You are an expert in creating engaging video summaries of research papers.
        Create a script for a 1-2 minute video summary of this paper:
        
        {paper_text[:10000]}  # Limit context length
        
        Your response should be a JSON with the following structure:
        {
            "title": "Video Title",
            "script": "Narrator text for the video...",
            "sections": [
                {
                    "start_time": 0,
                    "end_time": 15,
                    "image_prompt": "Description of the image to show during this section",
                    "narration": "Text to be spoken during this section"
                },
                ...
            ]
        }
        """
        
        # Call the LLM
        response = llm.invoke(prompt)
        
        try:
            # Parse the response
            video_script = json.loads(response.content)
            
            # Initialize video generator
            generator = VideoGenerator()
            
            # Generate the video
            result = generator.generate_video_from_script(video_script)
            
            if result['success']:
                # Get relative path for the web
                video_path = Path(result['video_path'])
                relative_path = str(video_path.relative_to(Path.cwd()))
                
                return jsonify({
                    "status": "completed",
                    "video_url": f"/static/videos/{video_path.name}",
                    "duration": result['duration'],
                    "resolution": result['resolution'],
                    "sections": len(video_script.get('sections', [])),
                    "title": video_script.get('title', 'Research Video')
                })
            else:
                return jsonify({
                    "error": f"Video generation failed: {result.get('error', 'Unknown error')}"
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
            "error": f"Failed to generate video: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500


# 🔍 Get paper info
@app.route("/paper/<paper_id>")
def get_paper_info(paper_id):
    """Get paper information including metadata required for the knowledge graph."""
    path = os.path.join(CACHE_DIR, f"{paper_id}.pdf")
    if not os.path.exists(path):
        return None
    
    try:
        # Get file metadata
        file_size = os.path.getsize(path)
        file_size_mb = round(file_size / (1024 * 1024), 2)
        
        # In a real implementation, you would extract this information from the PDF metadata
        # or from your database. For now, we'll return placeholder data.
        return {
            "id": paper_id,
            "title": f"Research Paper {paper_id}",
            "abstract": "Abstract not available. This is a placeholder abstract for the knowledge graph.",
            "year": datetime.now().year,
            "citations": 0,
            "downloads": 0,
            "authors": [{"name": f"Author {i+1}"} for i in range(3)],  # Placeholder authors
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

@app.route("/api/knowledge-graph/author/<author_name>")
def get_author_network(author_name: str):
    """Get network for a specific author."""
    try:
        author_id = knowledge_graph._get_author_id({"name": author_name})
        network = knowledge_graph.get_author_network(author_id)
        return jsonify(network)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Initialize spaNGy model for knowledge graph
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

if __name__ == "__main__":
    app.run(debug=True, port=5000)
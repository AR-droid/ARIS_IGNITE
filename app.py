import os
import requests
import base64
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv

# local summarizer functions
from summarizer import generate_summaries, generate_summaries_streaming, chat_with_paper
from diagram_generator import (
    generate_diagram, 
    get_available_diagram_types, 
    analyze_content_for_diagram_suggestions,
    get_diagram_examples,
    get_model_info
)

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
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
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
                        r = requests.get(download_url, timeout=30)
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
@app.route("/summarize/<paper_id>")
def summarize_paper(paper_id):
    path = os.path.join(CACHE_DIR, f"{paper_id}.pdf")
    if not os.path.exists(path):
        return jsonify({"error": "Paper not found. Please search for the paper first."}), 404
    
    try:
        summaries = generate_summaries(path, paper_id=paper_id)
        return jsonify(summaries)
    except Exception as e:
        error_msg = str(e)
        if "Ollama" in error_msg or "connection" in error_msg.lower():
            return jsonify({"error": "AI service unavailable. Please ensure Ollama is running with the 'mistral' model."}), 503
        return jsonify({"error": "Summarization failed", "details": error_msg}), 500


# 📝 Summarize with RAG (Streaming - shows summaries as they're ready)
@app.route("/summarize-streaming/<paper_id>")
def summarize_paper_streaming(paper_id):
    path = os.path.join(CACHE_DIR, f"{paper_id}.pdf")
    if not os.path.exists(path):
        return jsonify({"error": "Paper not found. Please search for the paper first."}), 404
    
    def generate():
        try:
            for level, summary, status in generate_summaries_streaming(path, paper_id=paper_id):
                # Send Server-Sent Events (SSE) format
                data = {
                    "level": level,
                    "summary": summary,
                    "status": status,
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(data)}\n\n"
                
        except Exception as e:
            error_msg = str(e)
            if "Ollama" in error_msg or "connection" in error_msg.lower():
                error_data = {
                    "level": "error",
                    "summary": "AI service unavailable. Please ensure Ollama is running with the 'mistral' model.",
                    "status": "error",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                error_data = {
                    "level": "error",
                    "summary": f"Summarization failed: {error_msg}",
                    "status": "error",
                    "timestamp": datetime.now().isoformat()
                }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Cache-Control'
        }
    )


# 💬 Chat with RAG
@app.route("/chat/<paper_id>", methods=["POST"])
def chat_paper(paper_id):
    data = request.get_json() or {}
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Empty query"}), 400

    path = os.path.join(CACHE_DIR, f"{paper_id}.pdf")
    if not os.path.exists(path):
        return jsonify({"error": "Paper not found. Please search for the paper first."}), 404

    try:
        answer = chat_with_paper(path, paper_id=paper_id, query=query)
        return jsonify({"answer": answer})
    except Exception as e:
        error_msg = str(e)
        if "Ollama" in error_msg or "connection" in error_msg.lower():
            return jsonify({"error": "AI service unavailable. Please ensure Ollama is running with the 'mistral' model."}), 503
        return jsonify({"error": "Chat failed", "details": error_msg}), 500


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


@app.route("/model-info")
def get_model_info_endpoint():
    """Get information about the Hugging Face models"""
    try:
        info = get_model_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": "Failed to get model info", "details": str(e)}), 500


# 🔍 Get paper info
@app.route("/paper/<paper_id>")
def get_paper_info(paper_id):
    path = os.path.join(CACHE_DIR, f"{paper_id}.pdf")
    if not os.path.exists(path):
        return jsonify({"error": "Paper not found"}), 404
    
    try:
        # Get file size
        file_size = os.path.getsize(path)
        file_size_mb = round(file_size / (1024 * 1024), 2)
        
        return jsonify({
            "id": paper_id,
            "file_size_mb": file_size_mb,
            "cached": True
        })
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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
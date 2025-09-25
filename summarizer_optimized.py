import os
import pickle
import requests
from typing import Dict, List, Generator, Tuple, Optional, Any
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# directories for caching vectorstores
VSTORE_DIR = "vstores"
os.makedirs(VSTORE_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using PyPDF2."""
    text_chunks = []
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    except Exception as e:
        print(f"[extract_text_from_pdf] error: {e}")
        raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    if not text_chunks:
        raise Exception("No text could be extracted from the PDF. The file might be corrupted or contain only images.")
    
    return "\n\n".join(text_chunks)

def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Split text into chunks with overlap."""
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(text)
    except Exception as e:
        print(f"[_chunk_text] error: {e}")
        # Fallback to simple splitting
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def _vstore_path_for(paper_id: str) -> str:
    """Get the path for a vector store."""
    return os.path.join(VSTORE_DIR, f"{paper_id}_vstore")

def build_or_load_vectorstore(text: str, paper_id: str):
    """
    Build or load a FAISS vectorstore for the given text.
    """
    vpath = _vstore_path_for(paper_id)
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        print(f"[build_or_load_vectorstore] embedding model error: {e}")
        raise Exception(f"Failed to load embedding model: {str(e)}")

    if os.path.exists(vpath):
        try:
            vstore = FAISS.load_local(vpath, embeddings, allow_dangerous_deserialization=True)
            print(f"[build_or_load_vectorstore] loaded existing vectorstore from {vpath}")
            return vstore
        except Exception as e:
            print(f"[build_or_load_vectorstore] load failed, rebuilding: {e}")

    # Build new vectorstore
    try:
        docs = []
        chunks = _chunk_text(text, chunk_size=800, overlap=150)
        print(f"[build_or_load_vectorstore] created {len(chunks)} text chunks")
        
        for i, c in enumerate(chunks):
            if c.strip():
                docs.append(Document(page_content=c, metadata={"chunk_id": i}))

        if not docs:
            raise Exception("No valid text chunks could be created from the PDF")

        vstore = FAISS.from_documents(docs, embeddings)
        
        try:
            vstore.save_local(vpath)
            print(f"[build_or_load_vectorstore] saved vectorstore to {vpath}")
        except Exception as e:
            print(f"[build_or_load_vectorstore] save failed: {e}")

        return vstore
        
    except Exception as e:
        print(f"[build_or_load_vectorstore] build error: {e}")
        raise Exception(f"Failed to build vectorstore: {str(e)}")

def _get_llm():
    """Initialize and return a Groq LLM client."""
    try:
        groq_api_key = os.getenv("GROK")
        if not groq_api_key:
            raise ValueError("GROK API key not found in environment variables")
            
        return ChatGroq(
            temperature=0.3,
            model_name="llama-3.3-70b-versatile",
            groq_api_key=groq_api_key,
            max_tokens=1024,
            timeout=30,
            streaming=True
        )
    except Exception as e:
        print(f"[_get_llm] Error initializing Groq LLM: {e}")
        raise Exception(f"Failed to initialize LLM: {str(e)}")

def generate_summaries_streaming(pdf_path: str, paper_id: str = None) -> Generator[Tuple[str, str, str], None, None]:
    """
    Generate summaries at different levels of complexity.
    Yields: (level, summary, status) where status is 'generating', 'complete', or 'error'
    """
    try:
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            yield ("error", "Could not extract text from PDF.", "error")
            return

        if not paper_id:
            paper_id = os.path.splitext(os.path.basename(pdf_path))[0]

        vstore = build_or_load_vectorstore(text, paper_id)
        retriever = vstore.as_retriever(search_kwargs={"k": 3})

        # Optimized prompts for different summary levels
        prompts = {
            "scratch": {
                "prompt": """In 3-4 sentences, explain this research paper simply for a high school student:
                - What problem is being solved?
                - What is the main idea?
                - Why is it important?""",
                "max_tokens": 200
            },
            "undergrad": {
                "prompt": """In 5-6 sentences, summarize for an undergrad:
                - Research question and method
                - Key findings
                - Why it matters""",
                "max_tokens": 300
            },
            "masters": {
                "prompt": """In 6-8 sentences, provide a graduate-level summary:
                - Methodology
                - Key results
                - Implications""",
                "max_tokens": 400
            },
            "high": {
                "prompt": """In 8-10 sentences, provide expert analysis:
                - Methodological approach
                - Key contributions
                - Relation to field""",
                "max_tokens": 500
            }
        }

        for key, prompt_data in prompts.items():
            try:
                # Signal that we're starting this summary
                yield (key, "Generating...", "generating")
                
                print(f"[generate_summaries_streaming] generating {key} summary...")
                
                # Configure LLM with appropriate settings for this summary
                llm = _get_llm()
                llm.max_tokens = prompt_data["max_tokens"]
                
                # Create a new QA chain with the configured LLM
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=False
                )
                
                # Get the response
                response = qa.invoke({"query": prompt_data["prompt"]})
                
                # Extract and clean the response
                if isinstance(response, dict) and 'result' in response:
                    response_text = response['result']
                else:
                    response_text = str(response)
                
                response_text = response_text.strip()
                if response_text.startswith('"') and response_text.endswith('"'):
                    response_text = response_text[1:-1]
                
                yield (key, response_text, "complete")
                
            except Exception as e:
                error_msg = f"Error generating {key} summary: {str(e)}"
                print(f"[generate_summaries_streaming] {error_msg}")
                yield (key, error_msg, "error")
        
    except Exception as e:
        error_msg = f"Failed to generate summaries: {str(e)}"
        print(f"[generate_summaries_streaming] {error_msg}")
        yield ("error", error_msg, "error")

# Legacy function for backward compatibility
def generate_summaries(pdf_path: str, paper_id: str = None) -> Dict[str, str]:
    """Legacy function that collects all summaries into a dictionary."""
    summaries = {}
    for level, summary, status in generate_summaries_streaming(pdf_path, paper_id):
        if status == "complete":
            summaries[level] = summary
    return summaries

def chat_with_paper(pdf_path: str, paper_id: str = None, query: str = "") -> str:
    """
    Chat with the paper using RAG.
    """
    try:
        if not query.strip():
            return "Please provide a question about the paper."
            
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            return "Could not extract text from the PDF."
            
        if not paper_id:
            paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
            
        vstore = build_or_load_vectorstore(text, paper_id)
        retriever = vstore.as_retriever(search_kwargs={"k": 3})
        llm = _get_llm()
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        
        response = qa.invoke({"query": query})
        return response.get('result', str(response))
        
    except Exception as e:
        print(f"[chat_with_paper] Error: {str(e)}")
        return f"An error occurred while processing your question: {str(e)}"

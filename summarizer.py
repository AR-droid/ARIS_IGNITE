import os
import pickle
import requests
from typing import Dict, List, Generator, Tuple
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Updated LangChain imports to avoid deprecation warnings
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
    """Extract text from PDF using PyPDF2 (simple)."""
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
    """Return list of text chunks (words-based split). Optimized for faster processing."""
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
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks


def _vstore_path_for(paper_id: str) -> str:
    return os.path.join(VSTORE_DIR, f"{paper_id}_vstore")


def build_or_load_vectorstore(text: str, paper_id: str):
    """
    Build a FAISS vectorstore for text and persist it under VSTORE_DIR/paper_id.
    If it already exists, load and return it.
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
            # Fix pickle security warning by allowing dangerous deserialization for trusted local files
            vstore = FAISS.load_local(vpath, embeddings, allow_dangerous_deserialization=True)
            print(f"[build_or_load_vectorstore] loaded existing vectorstore from {vpath}")
            return vstore
        except Exception as e:
            print(f"[build_or_load_vectorstore] load failed, rebuilding: {e}")

    # build new vectorstore
    try:
        docs = []
        chunks = _chunk_text(text, chunk_size=800, overlap=150)  # Smaller chunks for faster processing
        print(f"[build_or_load_vectorstore] created {len(chunks)} text chunks")
        
        for i, c in enumerate(chunks):
            if c.strip():  # Only add non-empty chunks
                docs.append(Document(page_content=c, metadata={"chunk_id": i}))

        if not docs:
            raise Exception("No valid text chunks could be created from the PDF")

        vstore = FAISS.from_documents(docs, embeddings)
        
        try:
            vstore.save_local(vpath)
            print(f"[build_or_load_vectorstore] saved vectorstore to {vpath}")
        except Exception as e:
            print(f"[build_or_load_vectorstore] save failed: {e}")
            # Continue without saving - vectorstore is still usable

        return vstore
        
    except Exception as e:
        print(f"[build_or_load_vectorstore] build error: {e}")
        raise Exception(f"Failed to build vectorstore: {str(e)}")


def _get_llm():
    """
    Return a Groq LLM wrapper configured with Llama 3 model.
    Requires GROK environment variable to be set with the API key.
    """
    try:
        groq_api_key = os.getenv("GROK")
        if not groq_api_key:
            raise ValueError("GROK API key not found in environment variables")
            
        return ChatGroq(
            temperature=0.3,
            model_name="llama-3.3-70b-versatile",
            groq_api_key=groq_api_key,
            max_tokens=1024,  # Reduced for faster response
            timeout=30,       # Add timeout to prevent hanging
            streaming=True    # Enable streaming for better UX
        )
            
    except Exception as e:
        print(f"[_get_llm] Error initializing Groq LLM: {e}")
        raise Exception(f"Failed to initialize LLM: {str(e)}")


def generate_summaries_streaming(pdf_path: str, paper_id: str = None) -> Generator[Tuple[str, str, str], None, None]:
    """
    Generate summaries one by one and yield them immediately.
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
        llm = _get_llm()

        qa = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever,
            return_source_documents=False
        )

        # Shorter, more focused prompts for faster generation
        prompts = {
            "scratch": """Explain this research paper simply for a high school student. Focus on:
            - What problem does it solve?
            - What is the main idea?
            - Why is it important?
            Keep it under 150 words.""",
            
            "undergrad": """Summarize this research paper for an undergraduate student:
            - Research question and method
            - Key findings and results
            - Significance and limitations
            Keep it under 200 words.""",
            
            "masters": """Provide a detailed summary for a master's student:
            - Methodology and experimental design
            - Results and analysis
            - Implications and future work
            Keep it under 250 words.""",
            
            "high": """Expert analysis of this research paper:
            - Critical evaluation of methodology
            - Assessment of contributions
            - Comparison with existing work
            Keep it under 300 words."""
        }

        for key, prompt_data in prompts.items():
            try:
                # Signal that we're starting this summary
                yield (key, "Generating...", "generating")
                
                print(f"[generate_summaries_streaming] generating {key} summary...")
                
                # Get the prompt text and set max tokens
                prompt = prompt_data["prompt"]
                max_tokens = prompt_data.get("max_tokens", 500)
                
                # Configure the LLM with appropriate settings for this summary
                llm = _get_llm()
                llm.max_tokens = max_tokens
                
                # Create a new QA chain with the configured LLM
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=False
                )
                
                # Get the response
                response = qa.invoke({"query": prompt})
                
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


def generate_summaries(pdf_path: str, paper_id: str = None) -> Dict[str, str]:
    """
    Generate multi-level summaries using RAG (legacy function for backward compatibility):
      - scratch (high-school)
      - undergrad
      - masters
      - high (expert)
    Uses RetrievalQA over FAISS with local Ollama LLM.
    """
    try:
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            return {"error": "Could not extract text from PDF."}

        if not paper_id:
            # create a safe identifier
            paper_id = os.path.splitext(os.path.basename(pdf_path))[0]

        vstore = build_or_load_vectorstore(text, paper_id)
        retriever = vstore.as_retriever(search_kwargs={"k": 3})  # Reduced from 5 to 3 for faster processing
        llm = _get_llm()

        qa = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever,
            return_source_documents=False
        )

        # Shorter, more focused prompts for faster generation
        prompts = {
            "scratch": """Explain this research paper simply for a high school student. Focus on:
            - What problem does it solve?
            - What is the main idea?
            - Why is it important?
            Keep it under 150 words.""",
            
            "undergrad": """Summarize this research paper for an undergraduate student:
            - Research question and method
            - Key findings and results
            - Significance and limitations
            Keep it under 200 words.""",
            
            "masters": """Provide a detailed summary for a master's student:
            - Methodology and experimental design
            - Results and analysis
            - Implications and future work
            Keep it under 250 words.""",
            
            "high": """Expert analysis of this research paper:
            - Critical evaluation of methodology
            - Assessment of contributions
            - Comparison with existing work
            Keep it under 300 words."""
        }

        summaries = {}
        for key, prompt in prompts.items():
            try:
                print(f"[generate_summaries] generating {key} summary...")
                # Use invoke instead of run for newer LangChain
                response = qa.invoke({"query": prompt})
                
                # Extract the response content
                if isinstance(response, dict) and 'result' in response:
                    response_text = response['result']
                else:
                    response_text = str(response)
                
                # Clean up the response
                if response_text:
                    # Remove any extra whitespace and normalize
                    cleaned_response = " ".join(response_text.split())
                    summaries[key] = cleaned_response
                else:
                    summaries[key] = f"Failed to generate {key} summary"
                    
            except Exception as e:
                error_msg = f"Error generating {key} summary: {str(e)}"
                print(f"[generate_summaries] {error_msg}")
                summaries[key] = error_msg

        return summaries
        
    except Exception as e:
        error_msg = f"Failed to generate summaries: {str(e)}"
        print(f"[generate_summaries] {error_msg}")
        return {"error": error_msg}


def chat_with_paper(pdf_path: str, paper_id: str = None, query: str = "") -> str:
    """
    Run a RAG QA for an arbitrary query using the paper's vectorstore and Ollama.
    """
    if not query:
        return "Empty query."

    try:
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            return "Could not extract text from PDF."

        if not paper_id:
            paper_id = os.path.splitext(os.path.basename(pdf_path))[0]

        vstore = build_or_load_vectorstore(text, paper_id)
        retriever = vstore.as_retriever(search_kwargs={"k": 4})  # Reduced from 6 to 4 for faster processing
        llm = _get_llm()
        
        # Shorter, more focused prompt for faster responses
        enhanced_query = f"""Answer this question about the research paper: "{query}"
        
        Provide a clear, accurate answer based on the paper's content. Be specific and helpful.
        Keep your response concise and focused."""
        
        qa = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever,
            return_source_documents=False
        )

        response = qa.invoke({"query": enhanced_query})
        
        # Extract the response content
        if isinstance(response, dict) and 'result' in response:
            response_text = response['result']
        else:
            response_text = str(response)
        
        if not response_text:
            return "I couldn't generate a response. Please try rephrasing your question."
            
        # Clean up the response
        cleaned_response = " ".join(response_text.split())
        return cleaned_response
        
    except Exception as e:
        error_msg = f"Chat failed: {str(e)}"
        print(f"[chat_with_paper] {error_msg}")
        return error_msg

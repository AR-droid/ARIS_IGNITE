import os
import json
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import spacy
from collections import defaultdict
import numpy as np
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityType(str, Enum):
    PAPER = "PAPER"
    AUTHOR = "AUTHOR"
    METHOD = "METHOD"
    DATASET = "DATASET"
    METRIC = "METRIC"
    CONCEPT = "CONCEPT"
    INSTITUTION = "INSTITUTION"
    RESULT = "RESULT"

class RelationType(str, Enum):
    CITES = "CITES"
    AUTHORED_BY = "AUTHORED_BY"
    USES = "USES"
    COMPARES = "COMPARES"
    IMPROVES = "IMPROVES"
    INTRODUCES = "INTRODUCES"
    EVALUATES = "EVALUATES"
    DEMONSTRATES = "DEMONSTRATES"
    ACHIEVES = "ACHIEVES"
    BASED_ON = "BASED_ON"
    EXTENDS = "EXTENDS"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    PUBLISHED_IN = "PUBLISHED_IN"
    PRESENTS = "PRESENTS"
    ADDRESSES = "ADDRESSES"

class KnowledgeGraph:
    def __init__(self, data_dir: str = "knowledge_graph_data"):
        """Initialize the knowledge graph."""
        self.data_dir = data_dir
        self.graph = nx.DiGraph()
        
        # Try to load spacy model, but don't fail if not available
        self.nlp = None
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: en_core_web_sm model not found. Some NLP features will be disabled.")
            print("To enable full functionality, run: python -m spacy download en_core_web_sm")
        except Exception as e:
            print(f"Error loading spacy model: {e}")
        self.entity_types = [t.value for t in EntityType]
        self.relation_types = [r.value for r in RelationType]
        self._init_groq_client()
    
    def _init_groq_client(self):
        """Initialize the Groq client."""
        groq_api_key = os.getenv("GROK")
        if not groq_api_key:
            logger.warning("GROK API key not found. Some features may be limited.")
            self.llm = None
            return
        
        try:
            self.llm = ChatGroq(
                temperature=0.3,
                model_name="llama-3.3-70b-versatile",
                groq_api_key=groq_api_key,
                max_tokens=4000
            )
            logger.info("Successfully initialized Groq client")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            self.llm = None
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load existing graph if available
        self.load_graph()
    
    async def extract_entities_relationships(self, text: str) -> Dict:
        """Extract entities and relationships using Groq LLM."""
        if not self.llm:
            logger.warning("Groq client not initialized. Using basic extraction.")
            return self._basic_extraction(text)
            
        prompt = """You are an expert in NLP and knowledge graph construction. 
        Extract key entities and their relationships from the following research paper text.
        
        Return a JSON object with two lists:
        1. nodes: List of unique entities with 'id', 'type', and 'label' fields
        2. edges: List of relationships with 'source', 'target', 'type', and 'label' fields
        
        Entity types should be one of: METHOD, DATASET, METRIC, CONCEPT, RESULT
        Relationship types should be specific and descriptive (e.g., USES, COMPARES, IMPROVES, etc.)
        
        Paper text:
        """ + text[:10000]  # Limit context size
        
        try:
            response = await self.llm.ainvoke(prompt)
            result = json.loads(response.content)
            return result
        except Exception as e:
            logger.error(f"Error extracting entities with LLM: {e}")
            return self._basic_extraction(text)
    
    def _basic_extraction(self, text: str) -> Dict:
        """Basic entity and relationship extraction as fallback."""
        doc = self.nlp(text)
        entities = []
        
        # Simple noun chunk extraction as fallback
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1:  # Only multi-word phrases
                entities.append({
                    'id': f"ENT_{hash(chunk.text) & 0xffffffff:08x}",
                    'type': 'CONCEPT',
                    'label': chunk.text
                })
        
        return {
            'nodes': entities,
            'edges': []
        }
    
    async def add_paper(self, paper_id: str, title: str, abstract: str, 
                       authors: List[Dict], metadata: Optional[Dict] = None) -> None:
        """Add a paper to the knowledge graph with enhanced entity extraction."""
        # Add paper node
        self.graph.add_node(
            paper_id,
            type=EntityType.PAPER.value,
            label=title,
            abstract=abstract,
            added_at=datetime.utcnow().isoformat(),
            **(metadata or {})
        )
        
        # Add author nodes and relationships
        for author in authors:
            author_id = self._get_author_id(author)
            self.graph.add_node(
                author_id,
                type=EntityType.AUTHOR.value,
                label=author.get("name", ""),
                name=author.get("name", ""),
                affiliations=author.get("affiliations", [])
            )
            # Add AUTHORED_BY relationship
            self.graph.add_edge(
                paper_id, 
                author_id, 
                type=RelationType.AUTHORED_BY.value,
                label="authored by"
            )
            
            # Add affiliation relationships
            for affil in author.get("affiliations", []):
                affil_id = f"INST_{hash(affil) & 0xffffffff:08x}"
                self.graph.add_node(
                    affil_id, 
                    type=EntityType.INSTITUTION.value,
                    label=affil,
                    name=affil
                )
                self.graph.add_edge(
                    author_id, 
                    affil_id, 
                    type=RelationType.AFFILIATED_WITH.value,
                    label="affiliated with"
                )
        
        # Extract entities and relationships using LLM
        extraction = await self.extract_entities_relationships(f"{title}. {abstract}")
        
        # Add extracted entities
        entity_map = {}
        for entity in extraction.get('nodes', []):
            entity_id = entity.get('id', f"ENT_{hash(str(entity)) & 0xffffffff:08x}")
            entity_map[entity_id] = entity_id
            self.graph.add_node(
                entity_id,
                type=entity.get('type', 'CONCEPT'),
                label=entity.get('label', ''),
                name=entity.get('label', '')
            )
            # Link to paper
            self.graph.add_edge(
                paper_id,
                entity_id,
                type=RelationType.PRESENTS.value,
                label="presents"
            )
        
        # Add relationships between entities
        for rel in extraction.get('edges', []):
            source = rel.get('source')
            target = rel.get('target')
            if source in entity_map and target in entity_map:
                self.graph.add_edge(
                    entity_map[source],
                    entity_map[target],
                    type=rel.get('type', 'RELATED_TO'),
                    label=rel.get('label', 'related to')
                )
        
        self._save_graph()
    
    def add_citation(self, citing_paper_id: str, cited_paper_id: str) -> None:
        """Add a citation relationship between two papers."""
        if citing_paper_id in self.graph and cited_paper_id in self.graph:
            self.graph.add_edge(citing_paper_id, cited_paper_id, type="CITES")
            self._save_graph()
    
    def get_related_papers(self, paper_id: str, max_results: int = 10) -> List[Dict]:
        """Get papers related to a given paper."""
        if paper_id not in self.graph:
            return []
            
        # Get papers that share concepts with the given paper
        paper_concepts = {n for n in self.graph.successors(paper_id) 
                         if self.graph.nodes[n].get("type") == "CONCEPT"}
        
        related = defaultdict(int)
        for concept in paper_concepts:
            for paper in self.graph.predecessors(concept):
                if paper != paper_id and self.graph.nodes[paper].get("type") == "PAPER":
                    related[paper] += 1
        
        # Sort by number of shared concepts and return top results
        sorted_papers = sorted(related.items(), key=lambda x: -x[1])
        return [{"paper_id": pid, "shared_concepts": count, 
                 **self.graph.nodes[pid]} for pid, count in sorted_papers[:max_results]]
    
    def get_author_network(self, author_id: str, depth: int = 1) -> Dict:
        """Get co-authorship network for an author."""
        if author_id not in self.graph:
            return {}
            
        # Get co-authors (people who have co-authored papers with this author)
        co_authors = set()
        for paper in self.graph.predecessors(author_id):
            if self.graph.nodes[paper].get("type") == "PAPER":
                for author in self.graph.successors(paper):
                    if author != author_id and self.graph.nodes[author].get("type") == "AUTHOR":
                        co_authors.add(author)
        
        # Get papers by these authors
        papers = set()
        for author in list(co_authors) + [author_id]:
            papers.update([p for p in self.graph.predecessors(author) 
                         if self.graph.nodes[p].get("type") == "PAPER"])
        
        # Create subgraph
        nodes = list(papers) + list(co_authors) + [author_id]
        subgraph = self.graph.subgraph(nodes)
        
        return {
            "nodes": [{"id": n, **self.graph.nodes[n]} for n in nodes],
            "links": [{"source": u, "target": v, "type": d["type"]} 
                      for u, v, d in subgraph.edges(data=True)]
        }
    
    def get_concept_network(self, concept: str, max_depth: int = 2) -> Dict:
        """Get network of related concepts."""
        concept_id = f"CONCEPT_{hash(concept) & 0xffffffff:08x}"
        if concept_id not in self.graph:
            return {"nodes": [], "links": []}
        
        # Get papers mentioning this concept
        papers = [p for p in self.graph.predecessors(concept_id) 
                 if self.graph.nodes[p].get("type") == "PAPER"]
        
        # Get related concepts (concepts mentioned in the same papers)
        related_concepts = set()
        for paper in papers:
            for concept_node in self.graph.successors(paper):
                if (self.graph.nodes[concept_node].get("type") == "CONCEPT" and 
                    concept_node != concept_id):
                    related_concepts.add(concept_node)
        
        # Create network
        nodes = [{"id": concept_id, "name": concept, "type": "CONCEPT", "size": 10}]
        links = []
        
        for rc in related_concepts:
            rc_name = self.graph.nodes[rc].get("name", "")
            nodes.append({"id": rc, "name": rc_name, "type": "CONCEPT", "size": 8})
            links.append({"source": concept_id, "target": rc, "type": "RELATED_TO"})
        
        return {"nodes": nodes, "links": links}
    
    def save_graph(self, filename: Optional[str] = None) -> None:
        """Save the graph to a file."""
        if not filename:
            filename = os.path.join(self.data_dir, "knowledge_graph.json")
        
        # Convert graph to node-link format for serialization
        data = nx.node_link_data(self.graph)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_graph(self, filename: Optional[str] = None) -> bool:
        """Load the graph from a file."""
        if not filename:
            filename = os.path.join(self.data_dir, "knowledge_graph.json")
        
        if not os.path.exists(filename):
            return False
            
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data)
            return True
        except Exception as e:
            print(f"Error loading knowledge graph: {e}")
            return False
    
    def _save_graph(self) -> None:
        """Internal method to save the graph."""
        self.save_graph()
    
    def _get_author_id(self, author: Dict) -> str:
        """Generate a consistent ID for an author."""
        name = author.get("name", "").lower().strip()
        if not name:
            return f"AUTHOR_{hash(str(author)) & 0xffffffff:08x}"
        
        # Try to find existing author with the same name
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == "AUTHOR" and data.get("name", "").lower() == name:
                return node_id
        
        # If not found, create a new ID
        return f"AUTHOR_{hash(name) & 0xffffffff:08x}"
    
    def _extract_concepts(self, text: str, top_n: int = 10) -> List[str]:
        """Extract key concepts from text using NLP."""
        doc = self.nlp(text)
        
        # Extract noun chunks and named entities
        chunks = [chunk.text.lower() for chunk in doc.noun_chunks]
        entities = [ent.text.lower() for ent in doc.ents]
        
        # Combine and count occurrences
        all_terms = chunks + entities
        term_counts = defaultdict(int)
        for term in all_terms:
            # Skip short terms and common words
            if len(term.split()) > 1 or (len(term) > 3 and term.isalnum()):
                term_counts[term] += 1
        
        # Get top terms
        sorted_terms = sorted(term_counts.items(), key=lambda x: -x[1])
        return [term for term, _ in sorted_terms[:top_n]]
        
    def get_graph_data(self) -> Dict[str, List[Dict]]:
        """
        Get the complete graph data in a format suitable for the frontend.
        
        Returns:
            Dict[str, List[Dict]]: A dictionary with 'nodes' and 'links' keys containing the graph data
        """
        nodes = []
        links = []
        
        # Add all nodes
        for node_id, node_data in self.graph.nodes(data=True):
            node = {
                'id': str(node_id),
                'type': node_data.get('type', 'UNKNOWN'),
                'label': node_data.get('label', str(node_id)),
                'group': node_data.get('type', 'other').lower(),
                'size': 10,  # Default size
                **{k: v for k, v in node_data.items() if k not in ['type', 'label', 'group', 'size']}
            }
            
            # Adjust size based on node type
            if node['type'] == 'PAPER':
                node['size'] = 15
                node['title'] = node_data.get('title', '')
                node['abstract'] = node_data.get('abstract', '')
                node['year'] = node_data.get('year')
                node['citations'] = node_data.get('citations', 0)
                node['downloads'] = node_data.get('downloads', 0)
            elif node['type'] == 'AUTHOR':
                node['size'] = 12
                node['name'] = node_data.get('name', '')
            elif node['type'] == 'CONCEPT':
                node['size'] = 8
                node['description'] = node_data.get('description', '')
                
            nodes.append(node)
        
        # Add all links
        for source, target, data in self.graph.edges(data=True):
            link = {
                'source': str(source),
                'target': str(target),
                'type': data.get('type', 'RELATED_TO'),
                'value': data.get('weight', 1),
                **{k: v for k, v in data.items() if k not in ['type', 'weight']}
            }
            links.append(link)
        
        return {
            'nodes': nodes,
            'links': links
        }

# Global instance of the knowledge graph
knowledge_graph = KnowledgeGraph()

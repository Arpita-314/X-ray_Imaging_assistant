"""
Retrieval system for the scientific research assistant.
Handles document loading, embedding generation, and FAISS vector store operations.
"""

import os
import pickle
from typing import List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


class DocumentRetriever:
    """Handles document retrieval using FAISS vector store and sentence embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 500):
        """
        Initialize the document retriever.
        
        Args:
            model_name: Name of the sentence transformer model
            chunk_size: Size of text chunks for splitting documents
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Load documents from file paths.
        
        Args:
            file_paths: List of paths to text/markdown files
            
        Returns:
            List of Document objects
        """
        documents = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} does not exist")
                continue
                
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Create document with metadata
            doc = Document(
                page_content=content,
                metadata={"source": file_path, "filename": os.path.basename(file_path)}
            )
            documents.append(doc)
            
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of split Document objects
        """
        return self.text_splitter.split_documents(documents)
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """
        Create FAISS vector store from documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            FAISS vector store
        """
        # Split documents into chunks
        split_docs = self.split_documents(documents)
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(
            documents=split_docs,
            embedding=self.embeddings
        )
        
        return self.vectorstore
    
    def save_vectorstore(self, path: str):
        """Save vector store to disk."""
        if self.vectorstore is not None:
            self.vectorstore.save_local(path)
    
    def load_vectorstore(self, path: str):
        """Load vector store from disk."""
        self.vectorstore = FAISS.load_local(path, self.embeddings)
    
    def retrieve_documents(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant Document objects
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
            
        return self.vectorstore.similarity_search(query, k=k)
    
    def retrieve_with_scores(self, query: str, k: int = 3) -> List[tuple]:
        """
        Retrieve documents with similarity scores.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of (Document, score) tuples
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
            
        return self.vectorstore.similarity_search_with_score(query, k=k)


class RAGChain:
    """Combines retrieval and generation using LangChain."""
    
    def __init__(self, retriever: DocumentRetriever, llm):
        """
        Initialize RAG chain.
        
        Args:
            retriever: Document retriever instance
            llm: Language model for generation
        """
        self.retriever = retriever
        self.llm = llm
        
    def create_context(self, query: str, k: int = 3) -> str:
        """
        Create context from retrieved documents.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            Combined context string
        """
        docs = self.retriever.retrieve_documents(query, k=k)
        context_parts = []
        
        for i, doc in enumerate(docs):
            context_parts.append(f"Document {i+1}:\n{doc.page_content}\n")
            
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, context: str = None) -> str:
        """
        Generate response using retrieved context.
        
        Args:
            query: User query
            context: Optional pre-built context
            
        Returns:
            Generated response
        """
        if context is None:
            context = self.create_context(query)
        
        # Create prompt for the LLM
        prompt = f"""Based on the following context, answer the question about X-ray imaging and medical physics.

Context:
{context}

Question: {query}

Please provide a detailed and scientifically accurate answer based on the context provided. If the context doesn't contain enough information to fully answer the question, say so and provide what information you can.

Answer:"""

        # Generate response using the LLM
        try:
            response = self.llm(prompt, max_length=512, do_sample=True, temperature=0.7)
            if isinstance(response, list) and len(response) > 0:
                return response[0]['generated_text'].split("Answer:")[-1].strip()
            else:
                return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
        except Exception as e:
            return f"Error generating response: {str(e)}"


# Import torch here to avoid issues with model loading
import torch


def setup_retrieval_system(knowledge_files: List[str], vectorstore_path: str = "vectorstore") -> tuple:
    """
    Set up the complete retrieval system.
    
    Args:
        knowledge_files: List of paths to knowledge base files
        vectorstore_path: Path to save/load vector store
        
    Returns:
        Tuple of (retriever, rag_chain)
    """
    # Initialize retriever
    retriever = DocumentRetriever()
    
    # Check if vectorstore already exists
    if os.path.exists(vectorstore_path):
        print("Loading existing vector store...")
        retriever.load_vectorstore(vectorstore_path)
    else:
        print("Creating new vector store...")
        # Load and process documents
        documents = retriever.load_documents(knowledge_files)
        if not documents:
            raise ValueError("No documents loaded. Please check file paths.")
            
        # Create vector store
        retriever.create_vectorstore(documents)
        
        # Save for future use
        retriever.save_vectorstore(vectorstore_path)
        print(f"Vector store saved to {vectorstore_path}")
    
    # Load language model
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    llm = pipeline(
        "text-generation",
        model="google/flan-t5-small",
        device=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Create RAG chain
    rag_chain = RAGChain(retriever, llm)
    
    return retriever, rag_chain
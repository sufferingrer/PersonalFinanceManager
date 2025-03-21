import pandas as pd
from typing import List, Dict, Any, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from rag.document_store import DocumentStore

class RAGRetriever:
    """
    Retrieval-Augmented Generation system for financial data analysis
    and personalized recommendations.
    """
    def __init__(self, llm, document_store: DocumentStore):
        self.llm = llm
        self.document_store = document_store
        
    def retrieve_and_generate(self, query: str, financial_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Retrieve relevant documents and generate a response based on the query
        and available financial data.
        """
        # Update document store with financial data if provided
        if financial_data is not None and not financial_data.empty:
            self.document_store.add_financial_data(financial_data)
            
        # Retrieve relevant documents
        retrieved_docs = self.document_store.search(query, top_k=3)
        
        # Format retrieved documents for the prompt
        context = "\n\n".join([
            f"Документ {i+1}: {doc['title']}\n{doc['content']}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        # Create a prompt template for the response generation
        prompt_template = f"""
        На основе предоставленной информации о финансах пользователя и общих финансовых знаний, 
        дай подробный, информативный и полезный ответ на запрос пользователя.
        
        Контекст из базы знаний:
        {context}
        
        Запрос пользователя: {query}
        
        Твой ответ должен быть структурированным, понятным и полезным для пользователя.
        Дай конкретные рекомендации, основанные на предоставленных данных.
        """
        
        # Generate response using LLM
        response = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template("{prompt}")
        ).run(prompt=prompt_template)
        
        return {
            "query": query,
            "retrieved_documents": retrieved_docs,
            "response": response
        }

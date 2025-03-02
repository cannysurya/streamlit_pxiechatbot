import streamlit as st
from llama_index.readers.file import PDFReader
import nest_asyncio
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import SubQuestionQueryEngine 
from llama_index.core.tools import QueryEngineTool, ToolMetadata
import os
import openai

# Load API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key is None:
    st.error("⚠️ OpenAI API Key is missing! Set it as an environment variable. Open CMD, execute -> set OPENAI_API_KEY=<your OpenAI API key>")

reader = PDFReader()
pxie_4139 = reader.load_data("./data/pxie-4139_specifications.pdf")
pxie_4147 = reader.load_data("./data/pxie-4147_specifications.pdf")


nest_asyncio.apply()

# Convert documents to embeddings
pxie4139_index = VectorStoreIndex.from_documents(pxie_4139)
pxie4147_index = VectorStoreIndex.from_documents(pxie_4147)

pxie4139_query_engine = pxie4139_index.as_query_engine()
pxie4147_query_engine = pxie4147_index.as_query_engine()

query_engine_tools = [
    QueryEngineTool(
        query_engine=pxie4139_query_engine,
        metadata=ToolMetadata(name='pxie-4139', description='Provides information about pxie-4139 instrument')
    ),
    QueryEngineTool(
        query_engine=pxie4147_query_engine,
        metadata=ToolMetadata(name='pxie-4147', description='Provides information about pxie-4147 instrument')
    )
]
s_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools)


st.title("Product Specification Chatbot")
st.caption("You can switch to chat mode through navigation bar - Chat Bot")
query = st.text_input("Ask about PXIe-4139 or PXIe-4147:")

if query:
    response = s_engine.query(query)  # Change based on product selection
    st.write("**Answer:**", response.response)
    st.write("**Source:**", response)
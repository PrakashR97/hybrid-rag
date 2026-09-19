import streamlit as st
import requests
import os
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# --- Page Configuration ---
st.set_page_config(
    page_title="Hybrid RAG Studio", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for minor tweaks
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8633/8633215.png", width=60)
    st.title("Document Setup")
    st.markdown("Upload a document to initialize the Vector DB and Knowledge Graph.")
    
    with st.form("upload_form", clear_on_submit=False):
        file_type_choice = st.selectbox(
            "Select File Type",
            options=["PDF Document (.pdf)", "Word Document (.docx, .doc)", "Text or CSV (.txt, .csv)", "Excel (.xlsx, .xls)"]
        )
        
        if "PDF" in file_type_choice: allowed_extensions = ["pdf"]
        elif "Word" in file_type_choice: allowed_extensions = ["docx", "doc"]
        elif "Text" in file_type_choice: allowed_extensions = ["txt", "csv"]
        elif "Excel" in file_type_choice: allowed_extensions = ["xlsx", "xls"]
        
        uploaded_file = st.file_uploader("Choose a file", type=allowed_extensions)
        submitted = st.form_submit_button("🚀 Process Document", use_container_width=True)

        if submitted:
            if uploaded_file is not None:
                with st.spinner("Processing into FAISS & NetworkX..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")}
                    try:
                        response = requests.post(f"{BACKEND_URL}/upload", files=files)
                        if response.status_code == 200:
                            st.success(f"Successfully processed: {uploaded_file.name}")
                            st.session_state.messages = [] # Reset chat on new doc
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Connection failed: {e}")
            else:
                st.warning("Please upload a file before processing.")

    st.divider()
    
    # Status and Settings
    st.caption("Backend Status")
    try:
        requests.get(BACKEND_URL, timeout=1)
        st.markdown("🟢 Online")
    except:
        st.markdown("🔴 Offline (Check Uvicorn)")
        
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Main Layout ---
st.title("🧠 Hybrid RAG Studio")
st.markdown("Explore your document through conversational AI, semantic vector chunks, and entity knowledge graphs.")

# Initialize tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "🕸️ Knowledge Graph", "📚 Vector Chunks"])

# --- TAB 1: Chat Interface ---
with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.info("👋 Welcome! Please upload a document in the sidebar to begin chatting.")

    # Chat history container
    chat_container = st.container(height=500)
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat Input
    if query := st.chat_input("Ask a question about your document..."):
        # Display user message instantly
        with chat_container:
            with st.chat_message("user"):
                st.markdown(query)
                
        payload = {"query": query, "history": st.session_state.messages}
        st.session_state.messages.append({"role": "user", "content": query})

        # Display assistant response
        with chat_container:
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                try:
                    with requests.post(f"{BACKEND_URL}/query", json=payload, stream=True) as r:
                        if r.status_code == 200:
                            for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                                if chunk:
                                    full_response += chunk
                                    response_placeholder.markdown(full_response + "▌")
                            response_placeholder.markdown(full_response)
                            st.session_state.messages.append({"role": "assistant", "content": full_response})
                        else:
                            st.error("Upload a document first.")
                except Exception as e:
                    st.error("Connection failed.")

# --- TAB 2: Knowledge Graph ---
with tab2:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Entity Relationships")
        st.caption("Visualizing the factual triples extracted by the LLM.")
    with col2:
        refresh_graph = st.button("🔄 Refresh Graph", use_container_width=True)
        
    if refresh_graph:
        with st.spinner("Fetching graph data..."):
            try:
                resp = requests.get(f"{BACKEND_URL}/visualize/graph")
                if resp.status_code == 200:
                    data = resp.json()
                    if not data.get("nodes"):
                        st.info("Graph is empty. Upload a document first.")
                    else:
                        from pyvis.network import Network
                        net = Network(notebook=False, height="500px", width="100%", directed=True, bgcolor="#f8f9fa", font_color="#1E1E1E")
                        for node in data["nodes"]:
                            net.add_node(node, label=node, title=node, color="#3b82f6", size=15)
                        for edge in data["edges"]:
                            net.add_edge(edge["source"], edge["target"], title=edge["relation"], label=edge["relation"], color="#94a3b8")
                        
                        net.save_graph("graph.html")
                        with open("graph.html", 'r', encoding='utf-8') as HtmlFile:
                            source_code = HtmlFile.read()
                        components.html(source_code, height=550)
            except Exception as e:
                st.error("Could not fetch graph data.")
    else:
        st.info("Click 'Refresh Graph' to view the current Knowledge Graph.")

# --- TAB 3: Vector Chunks ---
with tab3:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("FAISS Document Store")
        st.caption("Visualizing the semantic chunks stored in the Vector Database.")
    with col2:
        refresh_vectors = st.button("🔄 Refresh DB", use_container_width=True)

    if refresh_vectors:
        with st.spinner("Fetching vector data..."):
            try:
                resp = requests.get(f"{BACKEND_URL}/visualize/vectors")
                if resp.status_code == 200:
                    data = resp.json()
                    total_chunks = data.get("total_chunks", 0)
                    chunks = data.get("chunks", [])
                    
                    if total_chunks == 0:
                        st.info("Vector DB is empty. Upload a document first.")
                    else:
                        st.metric(label="Total Semantic Chunks", value=total_chunks)
                        st.divider()
                        
                        for i, chunk_text in enumerate(chunks):
                            with st.expander(f"📄 Chunk {i+1}", expanded=(i==0)):
                                st.write(chunk_text)
            except Exception as e:
                st.error("Could not fetch vector data.")
    else:
        st.info("Click 'Refresh DB' to view the current Vector Store.")
import streamlit as st
from services.rag_with_vector_db import RagWithVectorDb

# Page Configuration
st.set_page_config(page_title="GDPR Compliance Expert", layout="wide")

# Initialize RAG Engine (Cached)
@st.cache_resource
def get_rag_engine():
    rag = RagWithVectorDb()
    rag.build_index()
    return rag

rag_engine = get_rag_engine()

# Sidebar Configuration
st.sidebar.title("Configuration")
top_k = st.sidebar.slider("Chunks to retrieve (Top K)", min_value=1, max_value=5, value=2)
top_recitals = st.sidebar.slider("Recitals to retrieve (Top K)", min_value=1, max_value=5, value=2)
# bool hybrid 
hybrid = st.sidebar.checkbox("Hybrid Search", value=True)

st.title("🛡️ GDPR Compliance Intelligence")
st.markdown("Chat with our AI expert to explore GDPR articles and supporting recitals.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If it's an assistant message with metadata, show the expander
        if "metadata" in message:
            with st.expander("Reference Details"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Retrieved Articles")
                    for i, chunk in enumerate(message["metadata"]["retrieved_chunks"], 1):
                        meta = chunk.get("metadata", {})
                        st.markdown(f"**[{i}] {meta.get('article_name', 'Article')}**")
                        st.caption(chunk["content"][:300] + "...")
                        
                with col2:
                    st.subheader("Preferred Recitals")
                    for i, recital in enumerate(message["metadata"]["preferred_recitals"], 1):
                        st.markdown(f"**{recital.split(':')[0]}**")
                        st.caption(recital.split(':', 1)[1].strip() if ':' in recital else recital)

# Chat Input
if prompt := st.chat_input("Ask a GDPR compliance question..."):
    # Clear visual feedback as soon as user submits
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate Answer
    with st.chat_message("assistant"):
        with st.spinner("Analyzing legal documentation..."):
            try:
                # Call the RAG query
                response_data = rag_engine.query(prompt, hybrid=hybrid, top_k=top_k, top_recitals=top_recitals)
                answer = response_data["answer"]
                retrieved_chunks = response_data["retrieved_chunks"]
                preferred_recitals = response_data["preferred_recitals"]

                # Display Answer
                st.markdown(answer)

                # Display Metadata in Expander
                with st.expander("Reference Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Retrieved Articles")
                        for i, chunk in enumerate(retrieved_chunks, 1):
                            meta = chunk.get("metadata", {})
                            st.markdown(f"**[{i}] {meta.get('article_name', 'Article')}**")
                            st.caption(chunk["content"][:300] + "...")
                            
                    with col2:
                        st.subheader("Preferred Recitals")
                        for i, recital in enumerate(preferred_recitals, 1):
                            st.markdown(f"**{recital.split(':')[0]}**")
                            st.caption(recital.split(':', 1)[1].strip() if ':' in recital else recital)

                # Save Response to History
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "metadata": {
                        "retrieved_chunks": retrieved_chunks,
                        "preferred_recitals": preferred_recitals
                    }
                })
            except Exception as e:
                st.error(f"An error occurred: {e}")
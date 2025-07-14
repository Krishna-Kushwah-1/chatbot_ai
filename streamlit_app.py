import streamlit as st
import requests
import time

st.set_page_config(page_title="Fast Chatbot", layout="wide")
st.title("⚡ Fast Document Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

tab1, tab2 = st.tabs(["📤 Upload Documents", "💬 Chat"])

with tab1:
    st.header("Upload Knowledge Files")
    uploaded_files = st.file_uploader(
        "Select PDF/TXT/DOCX/CSV files",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx", "csv"]
    )
    
    if st.button("Upload and Process"):
        if uploaded_files:
            files = [("files", (file.name, file, "application/octet-stream")) 
                    for file in uploaded_files]
            
            with st.spinner("Processing documents..."):
                response = requests.post(
                    "http://localhost:8000/upload_knowledge",
                    files=files
                )
                
            if response.status_code == 200:
                result = response.json()
                st.success(result["message"])
                for detail in result["details"]:
                    st.write(detail)
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        else:
            st.warning("Please select files to upload")

with tab2:
    st.header("Chat with Documents")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask about your documents..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Show immediate feedback
            message_placeholder.markdown("▌")
            
            try:
                # Start timer
                start_time = time.time()
                
                # Stream response
                with requests.post(
                    "http://localhost:8000/query",
                    data={"question": prompt},
                    stream=True
                ) as r:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            text = chunk.decode("utf-8")
                            full_response += text
                            message_placeholder.markdown(full_response + "▌")
                
                # Show final response
                message_placeholder.markdown(full_response)
                
                # Show timing info
                response_time = time.time() - start_time
                st.caption(f"Response time: {response_time:.2f} seconds")
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                message_placeholder.markdown(error_msg)
                full_response = error_msg
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
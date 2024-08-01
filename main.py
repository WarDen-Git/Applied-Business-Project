# main.py

import os
import streamlit as st
from utils.populate_database import (load_documents,
                                     split_documents,
                                     add_to_chroma)
from utils.query_data import query_rag





# ------------ Stateful Buttons ----------------
if 'update' not in st.session_state:
    st.session_state.update = False

if 'continue_update' not in st.session_state:
    st.session_state.continue_update = False
    
def update():
    st.session_state.update = True
    
def continue_update():
    st.session_state.continue_update = True


# ------------ Stateful Buttons ----------------


st.title('✅ Electronics Engineering Licensure Exam Generator')
with st.sidebar:
    intro = '''
    This Retrieval Augmentated Generation application is a project by
    Denver Magtibay from Smart Edge ECE Review Specialist.
    
    ### Be the BEST PREPARED ECE EXAMINEE 💯🙌
    '''
    st.write(intro)
    
    
    input_key = st.text_input('OpenAI API Key')
    os.environ["OPENAI_API_KEY"] = input_key
    
    if input_key:
        st.button("Update Database", on_click=update)
        if st.session_state.update:

            st.warning('Updating the database only works locally.')
            
            st.button("Continue", on_click=continue_update)
            if st.session_state.continue_update:
                with st.spinner("Updating Database"):
                    docs = load_documents()
                    chunks = split_documents(docs)
                    add_to_chroma(chunks)
            
            
if input_key:
    tos_topic = st.selectbox(
    "Select the topic for the licensure exam:",
    ("Signal Spectra and Processing",
        "Modulation",
        "Digital Communications",
        "Transmission and Antenna Systems",
        "Radiowave Propagation",
        "Data Communications")
    )

    mcq_num = st.number_input('How many items do you need?', step=1)
        
        

    if st.button("Generate Exam", type="primary"):
        # Generate exam questions
        with st.spinner("Generating Exam Questions"):
            output_mcq = query_rag(tos_topic, mcq_num)
        
        # Formatting the response
        questions_and_answers = output_mcq.content.split("### Answer Key:")
        questions = questions_and_answers[0].strip()
        answer_key = questions_and_answers[1].strip()

        st.write(questions)
        with st.sidebar.expander("See Answer Key"):
            st.write(answer_key)
        
        st.sidebar.success(output_mcq.response_metadata)
        st.sidebar.success(output_mcq.usage_metadata)
else:
    st.write("Please input API Key, select a topic and click the 'Generate Exam' button.")

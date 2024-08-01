import streamlit as st
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


from utils.embeddings import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
    As an Electronics Engineer Examiner, you are tasked with creating a licensure exam for future electronics engineers.
    You are to create with the following knowledge context in mind:
    
    {context}
    
    --- 
    Your exam will focus on the topic of {tos_topic}.
    

    #### Instructions:
    - This exam consists of {mcq_num} multiple-choice questions.
    - Each question has four answer choices: A, B, C, and D.
    - Choose the best answer for each question.
    - There is only one correct answer for each question.
    - Take random context everytime you receive this instruction and paraphrase the question to generate different output.
    
    Use the following format to output questions without an initial conversational response:

    ##### 1. question_1_text\n
    \na) question_1_choice_A
    \nb) question_1_choice_B
    \nc) question_1_choice_C
    \nd) question_1_choice_D

    ...

    ##### 50. question_50_text\n
    \na) question_50_choice_A
    \nb) question_50_choice_B
    \nc) question_50_choice_C
    \nd) question_50_choice_D
    
    Use the following format for the answers:
    ### Answer Key:\n
    1) answer) answer_text\n
    2) answer) answer_text\n
    ...
    50) answer) answer_text\n
    
    For example:
    
    ### Answer Key:
    1) a) answer_text
    2) b) answer_text
    ...
    50) c) answer_text

    """


def query_rag(tos_topic, mcq_num):
    
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(tos_topic, k=8)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, 
                                    tos_topic=tos_topic,
                                    mcq_num=mcq_num)
    # print(prompt)

    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\n\nSources: {sources}"
    print(formatted_response)
    return response_text

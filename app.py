import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

llm = CTransformers(
    model='models/llama-2-7b-chat.ggmlv3.q2_K.bin',
    model_type='llama',
    config={'max_new_tokens': 20, 'temperature': 0.1}
)

def generate_question_llama2(topic):
    template = """
                Generate a {topic}-related question.
                """
    prompt = PromptTemplate(input_variables=["topic"], 
                            template=template)
    response = llm(prompt.format(topic=topic))
    return response

def validate_answer_llama2(question, answer):
    template = """
            Question: {question}
            Answer: {answer}
            Validate the answer in "Good" or "Bad" only
            """
    prompt = PromptTemplate(input_variables=["question", "answer"], 
                            template=template)
    response = llm(prompt.format(question=question, answer=answer))

    model_answer_template = """
                                Question: {question}
                                Give Correct answer.
                            """
        
    model_answer_prompt = PromptTemplate(input_variables=["question"], template=model_answer_template)
        
    model_answer_response = llm(model_answer_prompt.format(question=question))
    
    return response,model_answer_response

st.set_page_config(page_title="AI application for Questions and Answers",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("AI application for Que Ans🤖")

topic = st.selectbox("Select a Topic", ['Geography', 'Health', 'Sports'])

if st.button("Generate Question"):
    question = generate_question_llama2(topic)
    st.session_state["generated_question"] = question
    st.write("Generated Question: ",question)

if "generated_question" in st.session_state:
    user_answer = st.text_input("Your Answer")
    
    if st.button("Validate Answer"):
        if user_answer:
            validation_result, model = validate_answer_llama2(st.session_state["generated_question"], user_answer)
            st.write("Validation Result: ",validation_result)
            st.write("Model Answer: ",model)
        else:
            st.warning("Error.")

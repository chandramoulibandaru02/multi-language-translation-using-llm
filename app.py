import os
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from tenacity import retry,stop_after_attempt,wait_fixed

load_dotenv()

api=os.getenv("GORQ_API_KEY")

model1="llama-3.1-8b-instant"
model2="llama-3.3-70b-versatile"


def load_model(model_name):
    return ChatGroq(
        model=model_name,
        temperature=0.5,
        api_key=api
    )

model=load_model(model1)


prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a professional multilingual translator. "
     "Translate all input strictly into {language}. "
     "Preserve meaning and tone. "
     "Do not add explanations."),
    ("user", "{text}")
])

chain=prompt|model

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def safe_invoke(data):
    return chain.invoke(data)

st.title("Multilingual Translation System")
text=st.text_input("Enter the text that you want to translate ")
target_language=st.text_input("Enter the language you want to translate")

if st.button("Translate"):
    if not text.strip():
        st.write("Please enter the text to translate")
    elif not target_language.strip():
        st.write("Please enter the language name to translate")
    else:

        try:
            start_time=time.time()
            response=safe_invoke({
                "text":text,
                "language":target_language
            })  
            latency=round(time.time()-start_time,3)
            translated_text=response.content.strip()


            st.subheader("Translated output")
            st.write(translated_text.content)

            st.write("Latency: {latency} seconds")
        except:
            
            

            



            st.warning("Primary model failed. switching to a fallback model......")

            fallback_model=load_model(model2)
            fallback_chain=prompt | fallback_model
            resp=fallback_chain.invoke(
                {
                    "text":text,
                    "language":target_language
                }
            )

            

            latency=round(time.time()-start_time,3)
            st.write(f"Latency: {latency} seconds")
            st.subheader(
            "Translated language"
            )
            st.write(resp.content)
            token_data = resp.response_metadata["token_usage"]
            prompt_tokens = token_data["prompt_tokens"]
            completion_tokens = token_data["completion_tokens"]
            total_tokens = token_data["total_tokens"]
            st.subheader("Token Usage")
            st.write(f"Prompt Tokens: {prompt_tokens}")
            st.write(f"Completion Tokens: {completion_tokens}")
            st.write(f"Total Tokens: {total_tokens}")

            
            





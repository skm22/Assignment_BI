import spacy
import wikipediaapi
import streamlit as st
import spacy_streamlit


# Title of the app
st.title("Text Preprocessing and NER")

# scrapping data from the wikkipedia for a given name or title
wiki = wikipediaapi.Wikipedia(language='en',
                            extract_format=wikipediaapi.ExtractFormat.WIKI)
user_input = st.text_input("Write over here what you want to search ", "Cricket")
user_text = wiki.page(user_input).text

# getting the entity of the all words 
# from the scrapped data and visualising on streamlit
nlp = spacy.load("en_core_web_sm")
res = nlp(user_text)
spacy_streamlit.visualize_ner(res, labels=nlp.get_pipe('ner').labels)
# ------------------------Import necessary libraries-------------------
import streamlit as st
import logging
import tempfile
import os
import matplotlib.pyplot as plt
import pandas as pd
from modules.notes_maker.notes_maker import make_notes_from_image
from modules.text_to_audio.text_to_audio import convert_text_to_audio
from modules.general_chatting.chat import return_chat
from modules.stock_market_sentiment.stock_sentiment import analyze_stock_sentiment
from modules.stock_market_sentiment.name_extractor import extract_company_name
from modules.gmail.gmail_main import gmail_operation
import base64
from intent_classifier.main import classify_intent
from modules.gmail.sub_intent_classifier.gmail_sub_intent_classifier import predict_sub_intent
from modules.voice_summary.main import summarize_audio
from modules.weather.weather_fetcher import get_weather
from modules.NL2SQL.query_generator import get_sql_from_ollama
from modules.NL2SQL.fetch_youtube_data import handle_youtube_query

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

st.set_page_config(page_title="AI Agent", layout="centered")

# ---------------------------------asthetics-------------------------------------------
st.sidebar.markdown("**[🔗 LinkedIn- Jay Vijay Sawant](https://www.linkedin.com/in/jay-sawant-0011/)**", unsafe_allow_html=True)
st.sidebar.markdown("**[🔗 LinkedIn- Jeswin Thomas](https://www.linkedin.com/in/jeswin-thomas-09446a1a2/)**", unsafe_allow_html=True)

def set_bg_from_local(img_path):
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
        css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

bg_folder = "backgrounds"
bg_files = [f for f in os.listdir(bg_folder) if f.endswith((".jpg", ".jpeg", ".png"))]

default_image = "Default.jpg"

if default_image in bg_files:
    bg_files.remove(default_image)
    bg_files.insert(0, default_image)
bg_files.insert(0, "None")

selected_bg = st.sidebar.selectbox("Choose Background", bg_files, index=1)

if selected_bg != "None":
    image_path = os.path.join(bg_folder, selected_bg)
    set_bg_from_local(image_path)

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    classifier_type = st.selectbox(
        "Select Intent Classifier",
        ["ml", "rule_based", "transformer", "llm"],
        index=0,
        help="Choose the method for intent classification"
    )
    

    


st.title("Chat with Multi-Purpose AI Agent")
st.caption("Built from scratch by Jay & Jeswin 🔥")


# -----------------------chat------------------

with st.form("chat_form"):
    prompt = st.text_input("Enter your message:")
    uploaded_file = st.file_uploader("Optional Attachment (Image/Audio/txt/document)", type=["jpg", "jpeg", "png", "mp3", "wav", "m4a", 'txt', 'docx'])
    submit = st.form_submit_button("Send")

if submit and prompt:
    st.markdown(f"### You: {prompt}")

    try:
        with st.spinner("Classifing user intent"):
            intent = classify_intent(prompt, method=classifier_type)

        st.markdown(f"_Intent Detected: `{intent}`_")
    except Exception as e:
        st.error("Intent classification failed.")
        logging.error("Error during intent classification", exc_info=True)
        intent = None

    if intent == "make_notes":
        st.subheader("Note Maker from Image")
        if uploaded_file and uploaded_file.type.startswith("image/"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
                    tmp_img.write(uploaded_file.read())
                    image_path = tmp_img.name

                notes = make_notes_from_image(image_path)
                st.text_area("Extracted Notes", value=notes, height=200)

                audio_path = "notes_audio.mp3"
                convert_text_to_audio(notes, audio_path)
                st.audio(audio_path)
            except Exception as e:
                st.error("Failed to extract notes or generate audio.")
                logging.error("Error in make_notes workflow", exc_info=True)
            finally:
                os.remove(image_path)
        else:
            st.error("Please upload a valid image file.")

    elif intent == 'nl2sql':
        st.subheader("NL2SQL")

        try:
            with st.spinner("Generating SQL query from your question..."):
                sql_query = get_sql_from_ollama(prompt)

            st.markdown("### Generated SQL Query")
            st.code(sql_query, language='sql')

            with st.spinner("Fetching data from Hive..."):
                df = handle_youtube_query(sql_query)

            if isinstance(df, pd.DataFrame):
                if df.empty:
                    st.warning("No data returned for this query.")
                else:
                    st.success("Query executed successfully!")
                    st.dataframe(df, use_container_width=True)


            else:
                st.error(f" Unexpected response from data fetcher:\n\n{df}")

        except Exception as e:
            st.error(" Error while processing NL2SQL query.")
            logging.error("NL2SQL intent handler failed", exc_info=True)


    elif intent == "convert_to_audio":
        st.subheader("Text to Speech")

        text_to_convert = ""

        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".txt"):
                    text_to_convert = uploaded_file.read().decode("utf-8")
                elif uploaded_file.name.endswith(".docx"):
                    import docx
                    doc = docx.Document(uploaded_file)
                    text_to_convert = "\n".join([para.text for para in doc.paragraphs])
                else:
                    st.warning("Unsupported file type. Please upload a .txt or .docx file.")
            except Exception as e:
                st.error("Could not read the uploaded file.")
                logging.error("File reading failed", exc_info=True)

        if not text_to_convert and prompt.strip():
            if "-" in prompt:
                text_to_convert = prompt.split("-", 1)[1].strip()
            else:
                st.warning("No hyphen found in prompt. Please use: convert to audio - your text")
        
        if text_to_convert:
            try:
                with st.spinner("Converting text to audio..."):
                    convert_text_to_audio(text_to_convert, "converted_audio.mp3")
                st.success("Audio generated!")
                st.audio("converted_audio.mp3")
            except Exception as e:
                st.error("Failed to convert text to audio.")
                logging.error("Text-to-audio conversion failed", exc_info=True)
        else:
            st.info("Please upload a .txt/.docx file or provide prompt in format: `convert to audio - your text here`.")

    elif intent == "stock_sentiment":
        st.subheader("Stock Market Sentiment Analysis")
        try:
            company_name, news_url = extract_company_name(prompt)
            if not company_name or not news_url:
                st.warning("Could not extract a valid company name. Try again.")
            else:
                with st.spinner(f"geting sentiment of {company_name}"):
                    df = analyze_stock_sentiment(news_url)
                
                if df.empty:
                    st.warning(f"No news articles found for `{company_name}`.")
                else:
                    sentiment_counts = df['sentiment'].value_counts().to_dict()
                    st.markdown("### Sentiment Distribution")
                    fig, ax = plt.subplots()
                    ax.pie(
                        sentiment_counts.values(),
                        labels=sentiment_counts.keys(),
                        autopct='%1.1f%%',
                        startangle=90
                    )
                    ax.axis('equal')
                    st.pyplot(fig)
                    st.success(f"News Sentiment for `{company_name}`")
                    st.dataframe(df)

        except Exception as e:
            st.error("Error while fetching sentiment.")
            logging.error("Stock sentiment analysis failed", exc_info=True)

    elif intent == "general_chat":
        st.subheader("General Chat")
        try:
            response = return_chat(prompt)
            st.success("Response:")
            st.write(response)
        except Exception as e:
            st.error("Failed to generate chat response.")
            logging.error("General chat error", exc_info=True)

    elif intent == "voice_summary":
        st.subheader("Audio Summarization")
        if uploaded_file and uploaded_file.type.startswith("audio/"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp_audio:
                tmp_audio.write(uploaded_file.read())
                audio_path = tmp_audio.name

            try:
                summary = summarize_audio(audio_path)
                st.text_area("Summarized Audio", value=summary, height=200)
            except Exception as e:
                st.error("Failed to summarize audio.")
                logging.error("Voice summary error", exc_info=True)
            finally:
                os.remove(audio_path)
        else:
            st.info("Please upload an audio file.")
    
    elif intent == "analyze_product_sentiment":
        st.subheader("Product Sentiment Analysis (Coming Soon)")
        st.info("Product sentiment analysis is not yet implemented.")
    
    elif intent == "get_weather":
        st.subheader("Weather")
        try:
            weather = get_weather(prompt)
            st.write(weather)
        except Exception as e:
            st.error("Failed to get weather")
            logging.error("get weather error", exc_info=True)

    elif intent == "gmail_operations":
        st.subheader("Gmail Operations")
        try:
            sub_intent = predict_sub_intent(prompt)
            st.markdown(f"_Gmail Sub-Intent: `{sub_intent}`_")
            result = gmail_operation(prompt)
            for idx, email in enumerate(result, 1):
                st.markdown(f"#### Email {idx}")
                st.markdown(f"- **From:** {email['From']}")
                st.markdown(f"- **Subject:** {email['Subject']}")
                st.markdown(f"- **Date:** {email['Date']}")
                st.markdown(f"- **Snippet:** {email['Snippet']}")
                st.text_area(f"Body of Email {idx}", email['Body'], height=150, key=f"email_body_{idx}")

            else:
                st.success(result)
        except Exception as e:
            st.error("Gmail operation failed.")
            logging.error("Gmail module failed", exc_info=True)

    else:
        st.warning("Unknown or unsupported intent.")


with st.container():
    st.markdown("---")
    with st.expander("What Can This AI Agent Do?", expanded=False):
        st.markdown("""
### **AI Agent Features**

#### Notes Maker
→ Generate notes from handwritten notebook page.

#### Text to Audio
→ Convert any text into audio.

#### General Chat
→ Talk to the AI like ChatGPT.
                    
#### Audio Summarization
→ Upload or record audio — the AI will transcribe and summarize it for you.


#### Stock Market Sentiment
→ Get market sentiment of a company/stock.

#### Gmail Operations
→ Perform tasks like:
- Get last N emails  
- Count unread emails  
- Search emails by keyword/sender  

#### Gmail Sub-Intent Classifier
→ Automatically detects sub-tasks from your Gmail-related queries.

---

*More features coming soon!*
""", unsafe_allow_html=True)

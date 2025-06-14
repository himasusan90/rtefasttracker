import streamlit as st
import requests
import re
import os
from bs4 import BeautifulSoup
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from typing import TypedDict
from gtts import gTTS


# 1. Configure Gemini API
# Make sure to replace "" with your actual API key
genai.configure(api_key="AIzaSyCzRZjBRehEeiMPAbkRVz-N4yrtQajCZ78")

# 2. Define state with all required fields
class ArticleState(TypedDict):
    url: str
    article_text: str
    source_sentences: list[str]
    radio_script_cited: str  
    linkedin_post_cited: str   
    twitter_post: str
    linkedin_post: str  # <-- ADD THIS
    radio_script: str 
    radio_script_ga: str 
    audio_path_en: str  # <-- ADD THIS
    audio_path_ga: str  # <-- ADD THIS

# 3. Implement node functions
def fetch_article(state: ArticleState):
    try:
        url = state["url"]
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        page = requests.get(url, headers=headers)
        page.raise_for_status()
        soup = BeautifulSoup(page.content, "html.parser")
        
        # --- UPDATED SELECTOR FOR ARTICLE BODY ---
        # The page uses the semantic <article> tag to wrap the main content.
        # This is a more robust selector than the previous class-based ones.
        article_element = soup.find('article')

        if article_element:
            headline = article_element.find('h1')
            body_paragraphs = article_element.find_all('p')
            
            headline_text = headline.get_text(strip=True) if headline else ""
            body_text = " ".join([p.get_text(strip=True) for p in body_paragraphs])
            
            if not body_text:
                raise ValueError("Found article tag, but it contains no paragraph text.")
            
            article_text = f"{headline_text}\n\n{body_text}"
            sentences = re.split(r'(?<=[.!?]) +', body_text)
        else:
            raise ValueError("Could not find the main <article> element on the page.")
        
        return {"article_text": article_text,"source_sentences": sentences}
    
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Network request failed: {str(e)}")
    except Exception as e:
        raise ValueError(f"Scraping failed: {str(e)}")



def generate_tweet(state: ArticleState):
    try:
        # NOTE: 'gemini-2.0-flash-exp' might be an experimental model name.
        # The standard model is 'gemini-1.5-flash'.
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            f"Create a concise Twitter post with relevant hashtags from this article (max 280 characters): {state['article_text'][:2000]}"
        )
        return {"twitter_post": response.text}
    except Exception as e:
        raise ValueError(f"Generation failed: {str(e)}")


def generate_linkedin_post(state: ArticleState):
    """Generates a LinkedIn post from the article text."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""Based on the following article, create a professional LinkedIn post suitable for a corporate audience. 
        Start with a strong hook to grab attention and end with an engaging question to encourage comments.
        
        Article: {state['article_text'][:2000]}"""
        
        response = model.generate_content(prompt)
        return {"linkedin_post": response.text}
    except Exception as e:
        raise ValueError(f"LinkedIn post generation failed: {str(e)}")

def generate_radio_script(state: ArticleState):
    """Generates a radio news script from the article text."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        numbered_sources = "\n".join([f"{i+1}. {sentence}" for i, sentence in enumerate(state['source_sentences'])])
        
        prompt = f"""
        You are a news writer. Your task is to write a 25-second radio news bulletin based ONLY on the following numbered source sentences.
        You must not use any information outside of this text.
        After each piece of information you include in the bulletin, you MUST cite the number of the source sentence it came from in the format [Source: number].

        Source Sentences:
        {numbered_sources}

        Radio Bulletin:
        """
        
        
        response = model.generate_content(prompt)
        return {"radio_script": response.text}
    except Exception as e:
        raise ValueError(f"Radio script generation failed: {str(e)}")
    
def translate_to_irish(state: ArticleState):
    """Translates the English radio script into broadcast-quality Irish."""
    try:
        # Check if there is a radio script to translate
        english_script = state.get("radio_script")
        if not english_script:
            return {"radio_script_ga": "No English script was generated to translate."}

        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""Translate the following English news script into formal, broadcast-quality Irish (Gaeilge).
        Ensure the tone is appropriate for a national news broadcast.

        English Script: "{english_script}"
        """
        
        response = model.generate_content(prompt)
        return {"radio_script_ga": response.text}
    except Exception as e:
        raise ValueError(f"Irish translation failed: {str(e)}")

def generate_audio_files(state: ArticleState):
    """Generates EN and GA audio files from the radio scripts."""
    if not os.path.exists("temp_audio"):
        os.makedirs("temp_audio")

    audio_path_en = None
    audio_path_ga = None

    # --- Generate English audio in its own try/except block ---
    try:
        english_script = state.get("radio_script")
        if english_script:
            path_en = "temp_audio/radio_en.mp3"
            tts_en = gTTS(english_script, lang='en', tld='ie')
            tts_en.save(path_en)
            audio_path_en = path_en
    except Exception as e:
        print(f"English audio generation failed: {e}")

    # --- Generate Irish audio in its own try/except block ---
    try:
        irish_script = state.get("radio_script_ga")
        if irish_script:
            # This line will fail as gTTS does not support 'ga'.
            # It is left here to demonstrate the issue.
            # Replace with a supported TTS library for Irish.
            path_ga = "temp_audio/radio_ga.mp3"
            tts_ga = gTTS(irish_script, lang='ga') 
            tts_ga.save(path_ga)
            audio_path_ga = path_ga
    except Exception as e:
        print(f"Irish audio generation failed as expected: {e}")
        
    return {"audio_path_en": audio_path_en, "audio_path_ga": audio_path_ga}

def process_citations(state: ArticleState):
    """Replaces citation placeholders with actual sentences and formats them."""
    try:
        raw_script = state.get("radio_script")
        sources = state.get("source_sentences")
        if not raw_script or not sources:
            return {"radio_script_cited": "Citation processing failed."}

        # Find all citation placeholders, e.g., [Source: 12]
        citation_markers = re.findall(r'\[Source: (\d+)\]', raw_script)
        
        # Remove the placeholders from the main script text
        clean_script = re.sub(r' ?\[Source: \d+\]', '', raw_script).strip()

        # Build a formatted list of cited sources
        cited_sources_text = ["\n\n---", "**Sources:**"]
        unique_source_indices = sorted(list(set([int(num) - 1 for num in citation_markers])))

        for index in unique_source_indices:
            if 0 <= index < len(sources):
                cited_sources_text.append(f"- \"{sources[index].strip()}\"")

        # Combine the clean script with its list of sources
        final_cited_script = clean_script + "\n" + "\n".join(cited_sources_text)
        
        return {"radio_script_cited": final_cited_script}
    except Exception as e:
        raise ValueError(f"Citation processing failed: {e}")


# 4. Build LangGraph workflow
workflow = StateGraph(ArticleState)
workflow.add_node("fetch_article", fetch_article)
workflow.add_node("generate_tweet", generate_tweet)
workflow.add_node("generate_linkedin", generate_linkedin_post) # <-- ADD THIS NODE
workflow.add_node("generate_radio", generate_radio_script) 
workflow.add_node("translate_to_irish", translate_to_irish) 
workflow.add_node("generate_audio", generate_audio_files)
workflow.add_node("process_citations", process_citations)


workflow.set_entry_point("fetch_article")
workflow.add_edge("fetch_article", "generate_tweet")
workflow.add_edge("fetch_article", "generate_linkedin")
workflow.add_edge("fetch_article", "generate_radio")
workflow.add_edge("generate_radio", "process_citations")
workflow.add_edge("process_citations", "translate_to_irish")
workflow.add_edge("translate_to_irish", "generate_audio")
app = workflow.compile()

# 5. Streamlit interface
st.set_page_config(layout="wide") # Use the full width of the page
st.title("RTÉ Fast-Track: AI Content Amplifier")
st.markdown("Enter an RTÉ News URL to automatically generate a complete, multi-platform content package.")

url = st.text_input("Paste RTÉ News article URL here:", label_visibility="collapsed", placeholder="Paste RTÉ News article URL here...")


if url:
    try:
        with st.spinner("Analyzing article and generating content package... Please wait."):
            # Execute the full workflow
            results = app.invoke({"url": url})
        
        st.divider() # Adds a nice horizontal line
        
        # Display Twitter and LinkedIn posts
        st.subheader("Social Media Posts")
        col1, col2 = st.columns(2)
        with col1:
            st.text_area("Twitter", results.get("twitter_post", "Generation failed."), height=150)
        with col2:
            st.text_area("LinkedIn", results.get("linkedin_post", "Generation failed."), height=150)

        st.divider()

        # Display Radio Scripts side-by-side
        st.subheader("Radio Bulletin Scripts & Audio")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Radio Script (EN) with Citations**")
            st.markdown(results.get("radio_script_cited", "Generation failed."))
            audio_path_en = results.get("audio_path_en")
            if audio_path_en and os.path.exists(audio_path_en):
                st.audio(audio_path_en)
        with col2:
            st.text_area("Radio Script (GA)", results.get("radio_script_ga", "Translation failed."), height=200)
            audio_path_ga = results.get("audio_path_ga")
            # As noted previously, gTTS does not support Irish ('ga'). 
            # This audio player will not appear until an alternative TTS service is used.
            if audio_path_ga and os.path.exists(audio_path_ga):
                        st.audio(audio_path_ga)
                        
# In your Streamlit UI section, under the Radio Scripts
           
            

                    # Keep the extracted article text in an expander for verification
        with st.expander("View Full Extracted Article Text"):
                     st.text(results.get("article_text", "No text extracted."))
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


import streamlit as st
import requests
import re
import os
from bs4 import BeautifulSoup
import google.generativeai as genai
from langgraph.graph import StateGraph
from typing import TypedDict
from gtts import gTTS
from moviepy.editor import *
from urllib.parse import urljoin

# --- 1. SECURE API CONFIGURATION ---
# Load API key from Streamlit secrets for security. This prevents crashes.
try:
   genai.configure(api_key="")
except (KeyError, FileNotFoundError):
    st.error("GEMINI_API_KEY not found. Please create a .streamlit/secrets.toml file and add your key.")
    st.stop()


# --- 2. STATE DEFINITION (Cleaned Up) ---
class ArticleState(TypedDict):
    url: str
    article_text: str
    source_sentences: list[str]
    twitter_post: str
    linkedin_post: str
    radio_script: str
    radio_script_cited: str
    radio_script_ga: str
    audio_path_en: str
    audio_path_ga: str # Kept for potential future use
    image_url: str
    image_path: str
    video_path: str


# --- 3. NODE FUNCTIONS (Unchanged from your file, assuming they are correct) ---
def fetch_article(state: ArticleState):
    """Fetches article text, sentences, and hero image."""
    try:
        url = state["url"]
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        page = requests.get(url, headers=headers)
        page.raise_for_status()
        soup = BeautifulSoup(page.content, "html.parser")

        article_element = soup.find('article')
        if not article_element:
            raise ValueError("Could not find the main <article> element on the page.")

        headline = article_element.find('h1')
        body_paragraphs = article_element.find_all('p')
        headline_text = headline.get_text(strip=True) if headline else ""
        body_text = " ".join([p.get_text(strip=True) for p in body_paragraphs])
        
        if not body_text:
            raise ValueError("Found article tag, but it contains no paragraph text.")
        
        article_text = f"{headline_text}\n\n{body_text}"
        sentences = re.split(r'(?<=[.!?]) +', body_text)

        image_url, image_path = None, None
        og_image_tag = soup.find("meta", property="og:image")
        if og_image_tag and og_image_tag.get("content"):
            image_url = og_image_tag["content"]
        else:
            first_img_tag = article_element.find("img")
            if first_img_tag and first_img_tag.get("src"):
                image_url = urljoin(url, first_img_tag["src"])
        
        if image_url:
            temp_dir = "temp_media"
            if not os.path.exists(temp_dir): os.makedirs(temp_dir)
            image_response = requests.get(image_url, headers=headers, stream=True)
            image_response.raise_for_status()
            image_path = os.path.join(temp_dir, "hero_image.jpg")
            with open(image_path, "wb") as f:
                for chunk in image_response.iter_content(8192): f.write(chunk)

        return {
            "article_text": article_text, "source_sentences": sentences,
            "image_url": image_url, "image_path": image_path
        }
    except Exception as e:
        raise ValueError(f"Scraping failed: {e}")

def generate_tweet(state: ArticleState):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(f"Create a concise Twitter post (max 280 chars) with hashtags from this article: {state['article_text'][:1500]}")
    return {"twitter_post": response.text}

def generate_linkedin_post(state: ArticleState):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"Create a professional LinkedIn post based on this article. Start with a hook, end with a question.\n\nArticle: {state['article_text'][:1500]}"
    response = model.generate_content(prompt)
    return {"linkedin_post": response.text}

def generate_radio_script(state: ArticleState):
    model = genai.GenerativeModel('gemini-1.5-flash')
    numbered_sources = "\n".join([f"{i+1}. {s}" for i, s in enumerate(state['source_sentences'])])
    prompt = f"Write a 25-second radio news bulletin based ONLY on these sources. Cite each claim with [Source: number].\n\nSources:\n{numbered_sources}\n\nBulletin:"
    response = model.generate_content(prompt)
    return {"radio_script": response.text}

def process_citations(state: ArticleState):
    raw_script, sources = state.get("radio_script"), state.get("source_sentences")
    if not raw_script or not sources: return {"radio_script_cited": "Citation processing failed."}
    
    markers = re.findall(r'\[Source: (\d+)\]', raw_script)
    clean_script = re.sub(r' ?\[Source: \d+\]', '', raw_script).strip()
    
    cited_texts = ["\n\n---", "**Sources:**"]
    unique_indices = sorted(list(set([int(n) - 1 for n in markers])))
    for index in unique_indices:
        if 0 <= index < len(sources):
            cited_texts.append(f"- \"{sources[index].strip()}\"")
    
    return {"radio_script_cited": clean_script + "\n".join(cited_texts)}

def translate_to_irish(state: ArticleState):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"Translate this English news script into formal, broadcast-quality Irish (Gaeilge): \"{state['radio_script']}\""
    response = model.generate_content(prompt)
    return {"radio_script_ga": response.text}

def generate_audio_files(state: ArticleState):
    if not os.path.exists("temp_media"): os.makedirs("temp_media")
    audio_path_en = None
    english_script = state.get("radio_script")
    if english_script:
        try:
            # Clean script for TTS
            clean_english_script = re.sub(r'\[Source: \d+\]', '', english_script)
            path_en = "temp_media/radio_en.mp3"
            tts_en = gTTS(clean_english_script, lang='en', tld='ie')
            tts_en.save(path_en)
            audio_path_en = path_en
        except Exception as e:
            print(f"English audio generation failed: {e}")
    return {"audio_path_en": audio_path_en}
from moviepy.editor import *
import re

def generate_audiogram(state: ArticleState):
    """
    Generates a robust video audiogram that handles different aspect ratios
    and enforces a maximum duration to prevent crashes.
    """
    try:
        image_path = state.get("image_path")
        audio_path = state.get("audio_path_en")
        script_text = state.get("radio_script")

        # Exit gracefully if any required asset is missing
        if not all([image_path, audio_path, script_text]):
            print("Skipping video generation: Missing image, audio, or script.")
            return {"video_path": None}

        # --- FIX 1: ENFORCE A MAXIMUM DURATION ---
        # Load the audio and cap its duration at 30 seconds for a "Short"
        audio_clip = AudioFileClip(audio_path)
        max_duration = 30
        clip_duration = min(audio_clip.duration, max_duration)
        audio_clip = audio_clip.subclip(0, clip_duration)

        # --- FIX 2: ROBUST IMAGE RESIZING & CROPPING ---
        # Define the target video size (portrait 9:16)
        video_size = (1080, 1920)
        
        # Load the image and resize it to fill the frame width, then crop vertically
        image_clip = (ImageClip(image_path)
                      .set_duration(clip_duration)
                      .resize(width=video_size[0]) # Resize to fit width
                      .crop(x_center=video_size[0]/2, y_center=video_size[1]/2, width=video_size[0], height=video_size[1]) # Crop to fit height
                      .set_position("center"))

        # --- SUBTITLE GENERATION (Improved for efficiency) ---
        subtitle_clips = []
        clean_script = re.sub(r'\[Source: \d+\]', '', script_text)
        words = clean_script.split()
        
        start_time = 0
        for word in words:
            # Stop adding words if we've exceeded the clip duration
            if start_time >= clip_duration:
                break
                
            duration = max(0.25, len(word) / 10.0)
            text_clip = (TextClip(word, 
                                fontsize=90, 
                                color='yellow', 
                                font='Arial-Bold',
                                stroke_color='black', 
                                stroke_width=3)
                         .set_position(('center', 0.8), relative=True)
                         .set_start(start_time)
                         .set_duration(duration))
            subtitle_clips.append(text_clip)
            start_time += duration

        # --- FINAL COMPOSITION ---
        final_clip = CompositeVideoClip([image_clip] + subtitle_clips, size=video_size)
        final_clip = final_clip.set_audio(audio_clip)
        
        video_path = "temp_media/audiogram_short.mp4"
        final_clip.write_videofile(video_path, codec="libx264", audio_codec="aac", fps=24)
        
        return {"video_path": video_path}
    
    except Exception as e:
        # Provide a more detailed error log in the terminal
        print(f"!!! Video generation failed with an exception: {str(e)}")
        return {"video_path": None}


# --- 4. BUILD AND COMPILE THE CORRECTED WORKFLOW ---
workflow = StateGraph(ArticleState)
workflow.add_node("fetch_article", fetch_article)
workflow.add_node("generate_tweet", generate_tweet)
workflow.add_node("generate_linkedin", generate_linkedin_post)
workflow.add_node("generate_radio", generate_radio_script)
workflow.add_node("process_citations", process_citations)
workflow.add_node("translate_to_irish", translate_to_irish)
workflow.add_node("generate_audio", generate_audio_files)
workflow.add_node("generate_audiogram", generate_audiogram)

workflow.set_entry_point("fetch_article")

# After fetching, run text generation tasks in parallel
workflow.add_edge("fetch_article", "generate_tweet")
workflow.add_edge("fetch_article", "generate_linkedin")
workflow.add_edge("fetch_article", "generate_radio")

# The main media pipeline must be sequential
workflow.add_edge("generate_radio", "process_citations")
workflow.add_edge("generate_radio", "translate_to_irish")
workflow.add_edge("translate_to_irish", "generate_audio")
workflow.add_edge("generate_audio", "generate_audiogram")

app = workflow.compile()


# --- 5. STREAMLIT UI (Unchanged) ---
st.set_page_config(layout="wide", page_title="AI Content Amplifier", page_icon="🚀")
st.title("🚀 RTÉ Fast-Track: AI Content Amplifier")
st.markdown("Enter an RTÉ News URL to automatically generate a complete, multi-platform content package.")

url = st.text_input("Paste RTÉ News article URL here:", label_visibility="collapsed", placeholder="Paste RTÉ News article URL here...")

if url:
    try:
        with st.spinner("Brewing your media package... This may take up to a minute."):
            results = app.invoke({"url": url})
        
        st.success("Content package generated successfully!")
        st.divider()
        
        st.subheader("📱 Social Media Posts")
        col1, col2 = st.columns(2)
        with col1:
            st.text_area("Twitter Post", results.get("twitter_post", "Generation failed."), height=150)
        with col2:
            st.text_area("LinkedIn Post", results.get("linkedin_post", "Generation failed."), height=150)

        st.divider()

        st.subheader("🎙️ Radio Bulletin & Audio")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Radio Script (EN) with Citations**")
            st.markdown(results.get("radio_script_cited", "Generation failed."))
            audio_path_en = results.get("audio_path_en")
            if audio_path_en and os.path.exists(audio_path_en):
                st.audio(audio_path_en)
        with col2:
            st.markdown("**Radio Script (GA)**")
            st.text_area("Radio Script (GA)", results.get("radio_script_ga", "Translation failed."), height=200, label_visibility="collapsed")
                        
        st.divider()

        st.subheader("🎬 The Audiogram Short")
        video_path = results.get("video_path")
        if video_path and os.path.exists(video_path):
            st.video(video_path)
            st.markdown("A short video for social media (TikTok, Reels, Shorts) with voiceover and dynamic subtitles.")
        else:
            st.warning("Video generation failed or was skipped. This can happen if no image was found or due to a processing error.")
            
        with st.expander("🔍 View Full Extracted Article Text for Verification"):
             st.text(results.get("article_text", "No text was extracted."))

    except Exception as e:
        st.error(f"An unexpected error occurred in the workflow: {e}")
        st.info("Please check your terminal logs for detailed error messages from the graph nodes.")

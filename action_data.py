"""
This script performs a complete, advanced extraction of immediate, actionable 
suggestions from the health coaching dialogue dataset.

It processes a directory of raw CSV conversation files, uses an LLM to identify 
and structure the actions, and outputs a clean, unified JSON dataset.
"""

import os
import glob
import pandas as pd
import asyncio
import aiohttp
import json
from tqdm.asyncio import tqdm_asyncio
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError
import hashlib
import dotenv
dotenv.load_dotenv()

INPUT_DIR = "./data/raw_with_predicted_labels"  # Directory containing the patient*.csv files
OUTPUT_FILE = "suggestive_actions_dataset.json"
CACHE_DIR = "llm_cache"
CONVERSATION_GROUPING_MINUTES = 5  # Group coach messages sent within this time window
LLM_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key="
LLM_API_KEY = os.getenv("GEMINI_API")

# --- Pydantic Models for Data Validation ---
# This ensures the LLM output conforms to our desired structure.
class ActionItem(BaseModel):
    action_title: str = Field(..., description="A short, clear title for the action (e.g., 'Go for a walk').")
    action_description: str = Field(..., description="The full description of what the user should do.")
    category: str = Field(..., description="The assigned category from the list provided.")

class LLMResponse(BaseModel):
    actions: List[ActionItem]

# --- LLM Prompt Engineering ---
# This is the master prompt that instructs the LLM on its task.
MASTER_PROMPT = """
You are a highly intelligent research assistant specializing in behavioral psychology. Your task is to analyze text from a health coach and extract ONLY immediate, actionable, self-help suggestions.

**CRITICAL RULES:**
1.  **Extract ONLY Immediate Actions:** The suggestion must be something the user can do right now or in the very near future (e.g., "take a walk," "breathe deeply," "drink a glass of water").
2.  **IGNORE Non-Actions:** You MUST ignore greetings ("Hi there"), follow-ups ("How did your walk go?"), encouragement ("Great job!"), questions ("What do you want to do?"), and long-term advice ("You should consider joining a gym").
3.  **Be Concise:** Capture the core action. Rephrase it slightly into a clear, standalone instruction if necessary.
4.  **Categorize the action:** Assign a category from this exact list: [Mindfulness, Physical Activity, Cognitive, Social, Environmental, Nutrition, General].
5.  **Output Format:** You MUST provide your answer as a valid JSON object following this exact schema: {{"actions": [{{"action_title": "...", "action_description": "...", "category": "..."}}]}}.
6.  If no immediate actions are found in the text, you MUST return an empty list: {{"actions": []}}.

**Example:**
Input Text: "It's great you're aiming for 10k steps! If you feel stressed at your desk, try to just stand up and stretch for 60 seconds. Also, remember to drink water."
Your Output:
{{"actions": [{{"action_title": "Desk Stretch", "action_description": "If you feel stressed at your desk, try to just stand up and stretch for 60 seconds.", "category": "Physical Activity"}}, {{"action_title": "Hydrate", "action_description": "Remember to drink water.", "category": "Nutrition"}}]}}

---
**Now, process the following text from the health coach:**

{text_chunk}
"""

def get_cache_key(text: str) -> str:
    """Creates a unique hash for a text chunk to use as a cache filename."""
    return hashlib.md5(text.encode()).hexdigest()

async def call_llm_async(session: aiohttp.ClientSession, text_chunk: str, pbar) -> Optional[List[dict]]:
    """
    Asynchronously calls the LLM API with caching and exponential backoff.
    """
    cache_key = get_cache_key(text_chunk)
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")

    # 1. Check cache first
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            pbar.update(1)
            return json.load(f)

    # 2. If not in cache, call the API
    prompt = MASTER_PROMPT.format(text_chunk=text_chunk)
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    full_url = f"{LLM_API_URL}{LLM_API_KEY}"
    
    max_retries = 5
    delay = 1.0
    for attempt in range(max_retries):
        try:
            async with session.post(full_url, json=payload) as response:
                response.raise_for_status()
                result_json = await response.json()
                
                # Extract the text content which should be our JSON
                content_text = result_json.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '{}')
                
                # Clean the response to ensure it's valid JSON
                cleaned_text = content_text.strip().replace('`', '')
                if cleaned_text.startswith('json'):
                    cleaned_text = cleaned_text[4:]

                data = json.loads(cleaned_text)

                # 3. Validate with Pydantic
                validated_data = LLMResponse(**data)
                actions = [item.dict() for item in validated_data.actions]

                # 4. Save to cache
                with open(cache_path, 'w') as f:
                    json.dump(actions, f)

                pbar.update(1)
                return actions

        except (aiohttp.ClientError, json.JSONDecodeError, ValidationError) as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"Error processing chunk after {max_retries} attempts: {e}\nChunk: {text_chunk[:100]}...")
                pbar.update(1)
                return None
    return None


def group_coach_utterances(df: pd.DataFrame, max_minutes_diff: int) -> List[str]:
    """
    Groups consecutive coach utterances into contextually relevant chunks.
    """
    coach_df = df[df['speaker'] == 'Coach'].copy()
    
    # Handle multiple datetime formats
    def parse_datetime(time_str):
        try:
            # Try format 1: 2019-06-03 10:16:29
            return pd.to_datetime(time_str, format='%Y-%m-%d %H:%M:%S')
        except ValueError:
            try:
                # Try format 2: 2019/8/1 11:43
                return pd.to_datetime(time_str, format='%Y/%m/%d %H:%M')
            except ValueError:
                # Fallback to pandas auto-parsing
                return pd.to_datetime(time_str)
    
    coach_df['time'] = coach_df['time'].apply(parse_datetime)
    coach_df = coach_df.sort_values('time')
    
    chunks = []
    current_chunk = ""
    last_time = None
    
    for _, row in coach_df.iterrows():
        if last_time is None or (row['time'] - last_time).total_seconds() / 60 < max_minutes_diff:
            current_chunk += row['utterance'] + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = row['utterance'] + " "
        last_time = row['time']
        
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

async def main():
    """Main function to orchestrate the entire extraction pipeline."""
    print("--- Starting Advanced Action Extraction Pipeline ---")

    # Ensure output directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 1. Load and combine all patient CSVs
    all_files = glob.glob(os.path.join(INPUT_DIR, "patient*.csv"))
    if not all_files:
        print(f"Error: No patient CSV files found in directory '{INPUT_DIR}'.")
        return
        
    print(f"Found {len(all_files)} patient files to process.")
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    # 2. Group coach utterances into meaningful chunks
    print("Grouping coach utterances into conversational chunks...")
    text_chunks = group_coach_utterances(df, CONVERSATION_GROUPING_MINUTES)
    print(f"Created {len(text_chunks)} unique text chunks to analyze.")

    # 3. Process all chunks concurrently with a progress bar
    all_actions = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        with tqdm_asyncio(total=len(text_chunks), desc="Extracting Actions") as pbar:
            for chunk in text_chunks:
                tasks.append(call_llm_async(session, chunk, pbar))
            
            results = await asyncio.gather(*tasks)

            for result in results:
                if result:
                    all_actions.extend(result)
    
    # 4. Save the final, unified dataset
    print(f"\nExtraction complete. Found {len(all_actions)} total actionable suggestions.")
    
    # Remove duplicates
    unique_actions = [dict(t) for t in {tuple(d.items()) for d in all_actions}]
    print(f"Removed duplicates, resulting in {len(unique_actions)} unique actions.")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(unique_actions, f, indent=4)

    print(f"✅ Successfully saved the final dataset to '{OUTPUT_FILE}'")


if __name__ == "__main__":
    # To run this script, you need to have the patient CSV files
    # in a sub-folder named 'raw_with_predict_labels'.
    # Example:
    # your_project/
    # |-- llm_action_extractor.py
    # |-- raw_with_predict_labels/
    # |   |-- patient3.csv
    # |   |-- patient4.csv
    # |   |-- ...
    
    asyncio.run(main())

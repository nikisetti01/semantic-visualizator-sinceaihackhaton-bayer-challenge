# This module is responsible for interacting with the LLM API to relay user inputs and receive responses
# The functions in this module are called by the frontend module, either via user input or triggered
# by a preceding step in the RAG pipeline.

import requests
import os
from typing import Tuple
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv('LLM_TOKEN')
URL_CHAT = 'https://not.the.real.url/api/chat/agent'
HEADERS = {
    'accept': 'application/json',
    'Authorization': f"Bearer {TOKEN}",
    'Content-Type': 'application/json',
    }
QUERY_ASSISTANT_ID = os.getenv('QUERY_ASSISTANT_ID')
ANALYSIS_ASSISTANT_ID = os.getenv('ANALYSIS_ASSISTANT_ID')

def _post_prompt_to_assistant(prompt: str, assistant_id: str) -> Tuple[str, str]:
    
    data = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "assistant_id": assistant_id
    }

    response = requests.post(URL_CHAT, headers=HEADERS, json=data)
    if not response.status_code == 200:
        raise ValueError(f"Error: {response.content}")
    return prompt, response

def get_generated_query(user_input: str, session_state: dict) -> str:

    prompt = user_input
    prompt = f"These are the Requirements for the columns that are needed: {prompt}"
   
    if session_state.filters:
        prompt += f"""\n\nIncorporate these filters as additional WHERE clauses in the
            returned query, if not already specified in the base query: {str(session_state.filters)}""" 

    if session_state.hard_limit:
        prompt += f"""\n\nFinally, limit the number of returned observations to this number,
            regardless of what you have been previously instructed. Zero means no limit: {str(session_state.hard_limit)}"""

    _, response = _post_prompt_to_assistant(prompt, QUERY_ASSISTANT_ID)

    response_text = (response.json()['choices'][0]['message']['content']).strip("`").replace("sql\n", "").strip()
    return response_text, prompt

def analyze_observations(observations_sorted, question, columns, rows):
    prompt = f"""These are the {rows} needed Observations/Information ({columns}) to answer the 
        following question: {question}\n{observations_sorted}"""
    
    _, response_chat = _post_prompt_to_assistant(prompt, ANALYSIS_ASSISTANT_ID)
    try:
        response_text = (response_chat.json()['choices'][0]['message']['content']).strip("`").replace("sql\n", "").strip()
    except ValueError:
        response_text = None
    return response_text


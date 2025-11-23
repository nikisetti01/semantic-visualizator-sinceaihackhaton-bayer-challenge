import json
from from_text_to_streamlit_app.utils import *
from from_text_to_streamlit_app.prompts.text_to_json_prompt import get_text_to_json_prompt
from models.llm_client import OpenAILLMClient

def text_to_streamlit_app():

    client = OpenAILLMClient(
        model_name="gpt-4.1-mini",   # or "gpt-4.1-mini", "gpt-4.1-preview", etc.
        temperature=0.0,
        max_output_tokens=1500
    )
    
    datasets = from_csv_to_dict()

    prompt = get_text_to_json_prompt(datasets)

    response = client.invoke(prompt)

    print(response)

    cleaned_response = clean_response(response)
    
    workflow = json.loads(cleaned_response)
    json_to_streamlit(workflow, data_sources=datasets)
    

if __name__ == "__main__":
    text_to_streamlit_app()
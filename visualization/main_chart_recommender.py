# main_chart_recommender.py

import os
from viz_recommender.services.lida_service import create_lida_manager, load_dataframe, summarize_dataframe
from viz_recommender.services.prompt_loader import load_user_query
from viz_recommender.services.file_io import save_text_file
from viz_recommender.services.chart_recommender import build_user_prompt, generate_chart_recommendation
from models.llm_client import OpenAILLMClient  




def main():
    # 1. Setup API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable not found.")

    # 2. Setup LLM client
    llm_client = OpenAILLMClient(
        model_name="gpt-4.1",   # or "gpt-4.1-mini", "gpt-4.1-preview", etc.
        temperature=0.0,
    )

    # 3. LIDA Manager
    lida_manager = create_lida_manager(api_key=api_key)

    # 4. Load Data
    # TODO: Change the path to your actual CSV file
    # for each file in the folder 
    csv_path = "aggregated_data_1.csv"
    try:
        df = load_dataframe(csv_path)
    except FileNotFoundError:
        print(f"⚠️ File not found: {csv_path}")
        return

    # 5. LIDA Data Profiling
    print("📊 Generating data profile with LIDA...")
    data_profile_str = summarize_dataframe(df, lida_manager, summary_method="detailed")

    # 6. User Query
    try:
        user_query = load_user_query()
    except Exception as e:
        print(f"❌ Error loading user query: {e}")
        return


    # 7. Build full prompt (system + data profile + query)
    full_prompt = build_full_prompt(
        data_profile_str=data_profile_str,
        user_query=user_query,
    )

    # 8. Call LLM to get chart recommendations
    print("🧠 Analyzing user query with LLM...")
    text = generate_chart_recommendation(llm_client, full_prompt)

    # 9. Save output to file
    saved_path = save_text_file(text)
    print(f"\n✅ Analysis Complete. Saved to: {saved_path}")


if __name__ == "__main__":
    main()

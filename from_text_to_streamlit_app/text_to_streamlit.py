from openai import OpenAI
import pandas as pd
import json
from available_streamlit_components import SAFE_STREAMLIT_COMPONENTS
from from_text_to_streamlit_app.utils import *

def text_to_streamlit_app():
  datasets = {
      "incidents": pd.read_csv("data_1.csv"),
  }

  dataset_dicts = {name: df.to_dict(orient="records") for name, df in datasets.items()}

  best_visual = extract_best_visualization("chart_recomm_1.txt")

  schema = """
      {
        "components": [
          {
            "id": "unique string",
            "type": "Streamlit API element (e.g. markdown, metric, line_chart)",
            "args": {
              "data": ["list of positional args"],
              "config": {"keyword": "value", ...}
            },
            "dependencies": {
              "inputs": ["variables from previous components or datasets"],
              "outputs": ["variables this component produces"]
            },
            "layout": {
              "area": "main | sidebar | column",
              "column": "integer (optional)",
              "expander": "string (optional)"
            }
          }
        ]
      }
      """

  client = OpenAI(api_key = "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")
  
  prompt = f"""
      You are building a Streamlit workflow that shows summary statistics and visualization elements for the given datasets.
      Use ONLY the datasets provided below. You must NOT invent any new dataset names or variables. Reference datasets exactly as given.
      {json.dumps(dataset_dicts, indent=2)}

      You are already given the elements that must be shown in the Streamlit dashboard here:
      {best_visual}

      You must ONLY output Streamlit UI components listed in {SAFE_STREAMLIT_COMPONENTS}.

      For every component args MUST contain only "data" and "config". Nothing else must appear directly inside "args".
      args.data MUST contain ONLY dataset names ("incidents", "inspections") or workflow-state outputs.
      args.data MUST NEVER contain:
      - literal strings
      - numbers
      - dicts
      - arrays
      - constructed tables
      - fabricated values

      You are NOT allowed to construct ANY new tables, counts, aggregates, arrays, or rows manually.
      All data must come from datasets or from results of previous components.
      Use ONLY the types listed in {SAFE_STREAMLIT_COMPONENTS}.
      If you need to display a title or subtitle, use markdown with args.config.body.
      Do NOT invent any new component type names.

      WIDTH rules:
      - Allowed values: positive integers, "stretch"
      - Never use 0, negative numbers, or null.

      HEIGHT rules:
      - Allowed values: positive integers, "stretch"
      - Never use "content", 0, negative numbers, or null.

      For st.metric:
      - Do NOT include a "data" field.
      - Provide all parameters directly in "config": label, value, delta, help.
      - Do not pass a dataset as a positional argument.   
      - Use `args` as [label, value, delta]. 
      - Do NOT put label or value in config.
      - value must be a scalar (number or string). Do NOT pass a DataFrame or column reference directly.
      - If the value depends on a dataset, compute it beforehand and reference it via dependencies.inputs.
      - Never place `label`, `value`, or DataFrame inside `config` for metrics

      All columns used for plotting the y-axis in charts must be numeric. 
      If a column is categorical (like Severity with values Low/Medium/High), create a numeric version first (e.g., Severity_Numeric) and use that for the y-axis.
      Always include x and y parameters explicitly in chart configuration. 
      Do not rely on Streamlit to infer columns automatically.

      The "data" field must reference a dataset name from the provided datasets, or a valid table structure (list of dicts or 2D list). Do not output any strings that describe the data.

      Output just a JSON object following this schema: {schema}

      Each component can use outputs of previous components via dependencies.inputs. 

      All fields in the JSON must be valid JSON values: return raw JSON only, you must NOT include Python expressions, Pandas code, Jinja tempate variables, markdown, text, explainations or code fences.
      All keys and string values must be in double quotes, no trailing commas.
      Your JSON must reference only by name, not by code.
      You must NOT output any HTML, CSS, or inline styles anywhere in the JSON.
      Do NOT use unsafe_allow_html.
      Do NOT generate placeholders using HTML <div> or other tags.
      All non-data text must be plain strings inside markdown.

      Only include keys from the {SAFE_STREAMLIT_COMPONENTS} dictionary. 
      Do NOT invent additional parameters such as unsafe_allow_html unless they appear exactly in {SAFE_STREAMLIT_COMPONENTS}.
      """

  response = client.responses.create(
      model = "gpt-4.1-mini",
      input = prompt,
      temperature = 0
  )

  print(response.output_text)
  raw = response.output_text.strip()

  # Hard-clean markdown fences if present
  if raw.startswith("```"):
      raw = raw.strip("`")
      # Remove potential "json" specifier
      if raw.startswith("json"):
          raw = raw[len("json"):].strip()

  
  workflow = json.loads(raw)
  workflow = json.loads(response.output_text)
  workflow_state = json_to_streamlit(workflow, data_sources=datasets)
    

if __name__ == "__main__":
    text_to_streamlit_app()
import streamlit as st
import pandas as pd
from pathlib import Path


STREAMLIT_FRIENDLY_JSON_SCHEMA = """
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

RECOMMENDATION_OUTPUT_PATH = "chart_recommendation/recommendation.txt"

EXTRACTED_DATASETS_PATH = "datasets/extracted"

def from_csv_to_dict(datasets_path = EXTRACTED_DATASETS_PATH):
    datasets_path = Path(datasets_path)
    datasets = {}

    for csv_file in datasets_path.glob("*.csv"):
        dataset_name = csv_file.stem
        datasets[dataset_name] = pd.read_csv(csv_file)

    return datasets


def extract_best_visualization(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
    part2_started = False
    result_lines = []
    
    for line in lines:
        if "PART 2:" in line:
            part2_started = True
        if part2_started:
            result_lines.append(line)
    
    return ''.join(result_lines)


def resolve_data(json_component, workflow_state):
    args = json_component.get("args", {})
    
    if isinstance(args, dict):
        current_data = args.get("data", [])
    elif isinstance(args, list):
        current_data = args
    else:
        current_data = []

    resolved_data = []
    for d in current_data:
        if isinstance(d, str):
            if d in workflow_state:
                resolved_data.append(workflow_state[d])
            else:
                # invalid string → reject the whole component
                return None
        else:
            resolved_data.append(d)
    return resolved_data

def render_component(json_component, workflow_state, columns_map=None):
    # resolve layout
    layout = json_component.get("layout")
    current_streamlit_element = st

    if layout.get("area") == "sidebar":
        current_streamlit_element = st.sidebar
    
    col_idx = layout.get("column")
    if col_idx is not None and columns_map:
        current_streamlit_element = columns_map.get(col_idx, current_streamlit_element)
    
    # resolve current visualization element
    types = json_component.get("type", "").split(".")
    attribute = current_streamlit_element
    for type in types:
        attribute = getattr(attribute, type)

    # extract and validate data references
    resolved_data = resolve_data(json_component, workflow_state)
    if resolved_data is None:
        # skip component entirely if it references nonexistent datasets
        return
        
    # extract configuration for current element
    current_config = json_component.get("args", {}).get("config", {})
    resolved_config = {k: workflow_state.get(v, v) if isinstance(v, str) else v for k, v in current_config.items()}

    # add workflow dependencies if any
    for dependency in json_component["dependencies"].get("inputs", []):
        if dependency in workflow_state:
            current_config[dependency] = workflow_state[dependency]

    def call_attribute():
        try:
            return attribute(*resolved_data, **resolved_config)
        except TypeError:
            # fallback: try only first positional argument
            if resolved_data:
                return attribute(resolved_data[0], **resolved_config)
            else:
                return attribute(**resolved_config)
            
    # call the expander function if needed
    expander_title = layout.get("expander")
    if expander_title:
        with current_streamlit_element.expander(expander_title):
            output = call_attribute()
    else:
        output = call_attribute()

    # store element outputs if any (future dependency inputs)
    for var in json_component["dependencies"].get("outputs", []):
        workflow_state[var] = output


def json_to_streamlit(workflow, data_sources: dict[str, pd.DataFrame] = None):
    '''
    Dynamically executes the LLM-generated JSON workflow in the Streamlit app,
    without providig readable Python code
    '''
    if data_sources is None:
        data_sources = {}

    workflow_state = {**data_sources}

    # preprocess columns if any component uses the "column" layout
    max_col_idx = max(
        (comp.get("layout", {}).get("column", -1) for comp in workflow.get("components", [])),
        default=-1,
    )
    columns_map = {i: col for i, col in enumerate(st.columns(max_col_idx + 1))} if max_col_idx >= 0 else {}

    # render each component
    for comp in workflow.get("components", []):
        render_component(comp, workflow_state, columns_map)

    return workflow_state


def clean_response(response):
    raw = response.strip()
    # Hard-clean markdown fences if present
    if raw.startswith("```"):
        raw = raw.strip("`")
        # Remove potential "json" specifier
        if raw.startswith("json"):
            raw = raw[len("json"):].strip()

    return raw
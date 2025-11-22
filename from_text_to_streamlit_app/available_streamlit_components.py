# List of safe Streamlit components for LLM-driven JSON workflows

SAFE_STREAMLIT_COMPONENTS = {
    # Text & headings
    "title": ["body", "width"],
    "header": ["body", "width"],
    "subheader": ["body", "width"],
    "markdown": ["body", "width"],
    "caption": ["body", "width"],
    "text": ["body", "width"],
    "code": ["body", "language", "width", "height"],
    "divider": ["width"],

    # Metrics / simple outputs
    "metric": ["label", "value", "delta"],

    # Data outputs
    "dataframe": ["data", "width", "height"],
    "table": ["data"],

    # Simple charts
    "line_chart": ["data", "width", "height", "x", "y", "x_label", "y_label", "color"],
    "bar_chart": ["data", "width", "height", "x", "y", "x_label", "y_label", "color"],
    "area_chart": ["data", "width", "height", "x", "y", "x_label", "y_label", "color"],
    "scatter_chart": ["data", "width", "height", "x", "y", "x_label", "y_label", "color", "size"],
    "map": ["data", "zoom", "color", "size"],

    # Layout & containers
    "sidebar": [],
    "columns": ["spec", "width"],
    "container": ["width", "height"],
    "expander": ["label", "expanded", "width"],
    "empty": [],
    "tabs": ["tabs", "width"],

    # Utilities
    "spinner": ["text", "width"],
    "progress": ["value", "width", "text"],
}
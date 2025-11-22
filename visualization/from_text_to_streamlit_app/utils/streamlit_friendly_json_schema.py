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
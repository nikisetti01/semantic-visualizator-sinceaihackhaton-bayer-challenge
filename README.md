# 🚀 Insight Extraction & Auto-Visualization RAG Tool

This project implements a full **Insight Extraction + Visualization Pipeline** designed for the Bayer Oy challenge, on the occasion of the SinceAI Hackathon in Turku, Finland.  

Given:
- a **user prompt**
- a **dataset retrieved by a RAG system**,  

the system automatically produces:

- a **semantic intent JSON**
- enriched **categories** and semantic expansions
- embedding-based **categorical assignment**
- **SQL analytics** based on the user's prompt
- **visualization recommendations**
- a fully dynamic **Streamlit dashboard**

Everything runs **end-to-end** with a single pipeline.

---

# 🧩 System Pipeline Overview

![Pipeline Structure](pipeline.png)

The architecture is built around **three core modules**, all cooperating to turn natural language into structured analytics and visual insights.

---

# 1️⃣ Insight Extraction Module  
**Location:** `insight_extraction/`

This module interprets the user question and structurally organizes the dataset for downstream analysis.

### ✔ Semantic Intent
Transforms the natural-language prompt into a structured JSON containing:
- requested metrics  
- grouping dimensions  
- filters  
- semantic categories  
- logical operations  

This provides a clean, machine-readable blueprint for analytics generation.

### ✔ Semantic Category Expansion
For each category belonging to a dimension, the system expands it using an LLM:
- textual description  
- synonyms  
- example sentences  

These enrichments make semantic matching far more robust.

### ✔ Embedding-Based Category Assignment
We embed:
- dataset rows  
- expanded category descriptions  

Using cosine similarity, each row is assigned to the closest semantic category.

The result is a **new categorical dataset** ready for analytics.

### ✔ SQL Generator (Labelled SQL Blocks)
Using the enriched dataset, the module generates SQL queries in a consistent, parser-friendly format


Key guarantees:
- multiple, coherent insight queries  
- deterministic structure (`-- HEADER` labels)  
- full compatibility with our SQL executor  
- ready for visualization  

---

# 2️⃣ Visualization Recommender  
**Location:** `viz_recommender/`

Given:
- SQL-generated analytics DataFrames  
- the original user prompt  

the module selects the **best possible charts** to communicate insights:
- trends → line charts  
- category comparisons → bar charts  
- distributions → histograms  
- proportions → pie charts  
- anomalies → scatter/line hybrid  

It outputs a file:

recs.txt

containing high-level visualization instructions.

---

# 3️⃣ Streamlit Auto-Dashboard  
**Location:** `from_text_to_streamlit_app/`

A fully dynamic **Streamlit frontend** turns insights + recommendations into a live dashboard.
Instead of writing Streamlit code manually, the UI is created dynamically from an LLM-generated JSON workflow.
The JSON produced by the LLM specifies:
- UI layout (main, sidebar, columns, expanders)
- components (markdown, bar_chart, line_chart, caption, etc.)
- data inputs (referencing DataFrames)
- configuration (x/y encoding, labels, width/height, colors)
- dependencies and output chaining
The Streamlit renderer interprets each JSON block and dynamically calls the correct Streamlit element.

With a single command, the system:
- loads all analytic DataFrames (DF_1, DF_2, …)
- reads `recs.txt`
- renders each recommended chart
- supports multiple sections and layouts
- produces a polished analytical UI

# Final App
Run it with:

```bash
pip install -r requirements.txt
streamlit run app.py
▶️ Running the Full Pipeline


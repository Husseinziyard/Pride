# ⚡ DS Copilot — AI-Powered Data Science Assistant

An autonomous AI agent that assists with the complete data science pipeline through natural conversation. Upload a dataset and chat — it explores, cleans, visualizes, and models your data.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red?style=flat-square&logo=streamlit)
![LangGraph](https://img.shields.io/badge/LangGraph-ReAct_Agent-green?style=flat-square)
![OpenRouter](https://img.shields.io/badge/OpenRouter-Multi_Model-orange?style=flat-square)

## 🎯 What It Does

DS Copilot is a **ReAct (Reasoning + Acting) AI agent** that autonomously performs data science tasks. It reasons about what to do, selects the right tool, executes it, observes results, and iterates — all through a conversational interface.

### Core Capabilities

| Feature | Description |
|---------|-------------|
| 📊 **Exploratory Data Analysis** | Automatic overview — shape, types, missing values, statistics |
| 🧹 **Data Cleaning** | Handle missing values, drop duplicates, remove outliers, type conversion |
| 📈 **Visualization** | Histograms, scatter plots, bar charts, correlation heatmaps, box plots, line charts |
| 🤖 **ML Modeling** | Auto-detect task type, train Random Forest & Linear models, compare metrics |
| 💻 **Code Execution** | Write and run custom Python code directly on the dataset |
| 🔬 **Column Analysis** | Deep statistical analysis of individual columns |

## 🏗️ Architecture

```
User (Streamlit Chat UI)
        │
        ▼
   LangGraph ReAct Agent
   (Reason → Act → Observe → Repeat)
        │
        ├── LLM: OpenRouter (any model — Gemini, Claude, GPT, LLaMA, etc.)
        │
        ├── Memory: MemorySaver (conversation context across turns)
        │
        └── Tools:
            ├── dataset_overview()      → Full EDA
            ├── clean_data()            → Data cleaning operations
            ├── create_visualization()  → Charts & plots
            ├── build_model()           → Auto ML training & evaluation
            ├── execute_python_code()   → Run custom Python
            └── analyze_column()        → Deep column analysis
```

## 🚀 Quick Start

### 1. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/ds-copilot.git
cd ds-copilot
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 2. Add your API key (one-time setup)
```bash
cp .env.example .env
```
Edit `.env` and paste your [OpenRouter API key](https://openrouter.ai/keys):
```
OPENROUTER_API_KEY=sk-or-v1-your_actual_key
```

### 3. Run
```bash
streamlit run app.py
```

That's it. No API keys to paste every session — it loads from `.env` automatically.

## 📁 Project Structure

```
ds-copilot/
├── app.py              # Streamlit UI — professional chat interface
├── agent.py            # LangGraph ReAct agent with OpenRouter LLM
├── tools.py            # 6 data science tools (EDA, cleaning, viz, ML, code exec)
├── requirements.txt    # Python dependencies
├── .env.example        # API key template (copy to .env)
├── .gitignore
└── README.md
```

## 🛠️ Tech Stack

- **Agent Framework:** [LangGraph](https://github.com/langchain-ai/langgraph) — ReAct agent with memory
- **LLM Provider:** [OpenRouter](https://openrouter.ai/) — access 100+ models through one API
- **Default Model:** Google Gemini 2.0 Flash (fast, free tier available)
- **UI:** [Streamlit](https://streamlit.io/) — professional chat interface
- **Data:** pandas, NumPy, scikit-learn, matplotlib, seaborn

## 🔄 Changing the Model

Edit `.env` to use any model on OpenRouter:
```bash
DS_COPILOT_MODEL=anthropic/claude-sonnet-4    # Claude
DS_COPILOT_MODEL=openai/gpt-4o                # GPT-4o
DS_COPILOT_MODEL=meta-llama/llama-4-scout     # LLaMA 4
DS_COPILOT_MODEL=google/gemini-2.0-flash-001  # Gemini (default)
```

## 💡 Example Interactions

```
You: "Give me an overview of this dataset"
→ Agent calls dataset_overview() → returns shape, types, missing values, stats

You: "Fill missing values in Age with the median"
→ Agent calls clean_data(action="fill_missing", column="Age", strategy="median")

You: "Show me a correlation heatmap"
→ Agent calls create_visualization(chart_type="correlation")

You: "Build a model to predict Survived"
→ Agent calls build_model(target_column="Survived") → trains & compares models

You: "Write code to create a BMI feature from Height and Weight"
→ Agent calls execute_python_code() with custom pandas code
```

## 📋 Assignment Details

- **Course:** Data Science Applications and AI (LB3114)
- **Institution:** General Sir John Kotelawala Defence University
- **Year:** 3rd Year, 1st Semester — Intake 41

## 📄 License

Academic use.

"""
Pride — AI Agent
LangGraph ReAct agent powered by OpenRouter for data science tasks.
"""

import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from tools import ALL_TOOLS

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Change model if needed (see https://openrouter.ai/models)
MODEL_NAME = "meta-llama/llama-4-scout:free"


# ── System Prompt ──
SYSTEM_PROMPT = """You are Pride a DS Copilot expert AI Data Science Assistant.

You help users with the complete data science workflow through natural conversation.
You are like a senior data scientist sitting next to the user, guiding them step by step.

**How you work:**
1. When a user uploads data → use `dataset_overview` to understand it first
2. Proactively identify issues (missing values, wrong types, outliers)
3. Suggest and perform cleaning steps
4. Create meaningful visualizations to reveal patterns
5. Build and compare ML models when ready
6. Use `execute_python_code` for any custom analysis

**Your tools:**
- `dataset_overview` — Full EDA summary of the dataset
- `clean_data` — Handle missing values, duplicates, outliers, type conversion
- `create_visualization` — Create charts (histogram, scatter, bar, correlation, boxplot, line)
- `build_model` — Auto-train and compare ML models (classification or regression)
- `execute_python_code` — Run custom Python code on the dataset (df is available)
- `analyze_column` — Deep-dive into a specific column

**Rules:**
- ALWAYS use tools to interact with data. Never guess or fabricate numbers.
- After each action, explain what you found in plain language and suggest next steps.
- Keep responses clear and conversational — like a helpful colleague, not a textbook.
- When showing results, highlight the key insights and what they mean.
- If something is ambiguous, ask one clarifying question.
"""


def create_agent():
    """Create and return the DS Copilot agent."""
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "paste_your_key_here":
        raise ValueError(
            "API key not set. Open agent.py and paste your OpenRouter key "
            "in the OPENROUTER_API_KEY variable."
        )

    llm = ChatOpenRouter(
        model=MODEL_NAME,
        temperature=0.1,
        api_key=OPENROUTER_API_KEY,
    )

    memory = MemorySaver()

    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )

    return agent


def get_agent_response(agent, user_message: str, thread_id: str = "default"):
    """Get a response from the agent."""
    config = {"configurable": {"thread_id": thread_id}}

    response = agent.invoke(
        {"messages": [("user", user_message)]},
        config=config,
    )

    ai_messages = [
        msg for msg in response["messages"]
        if hasattr(msg, "type") and msg.type == "ai" and msg.content
    ]

    if ai_messages:
        return ai_messages[-1].content
    return "I processed your request but have no text response. Check if any charts were generated."

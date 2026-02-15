"""
Multi-Agent System for DocMind AI
Provides 6 specialized agents for different document interaction modes.
"""

from agno.models.openai import OpenAIChat
from agno.agent import Agent
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.tools.duckduckgo import DuckDuckGoTools
from typing import Dict, Any


AGENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "qa": {
        "name": "Q&A Assistant",
        "icon": "💬",
        "description": "Answers questions from your documents with precise citations and page references.",
        "instructions": [
            "You are a precise Q&A assistant for PDF documents.",
            "ALWAYS start by searching the knowledge base using search_knowledge_base tool.",
            "Provide exact citations with page numbers when available.",
            "Use bullet points and structured formatting for clarity.",
            "If information is not found in the document, clearly state this.",
            "Quote relevant passages directly from the source material.",
            "At the end of your response, suggest 1-2 related follow-up questions the user might find useful.",
        ],
    },
    "summarizer": {
        "name": "Summarizer",
        "icon": "📝",
        "description": "Generates structured, comprehensive summaries of document content.",
        "instructions": [
            "You are an expert document summarizer.",
            "Search the knowledge base thoroughly before summarizing.",
            "Create well-structured summaries with:",
            "  - Executive Summary (2-3 sentences)",
            "  - Key Points (bullet list)",
            "  - Detailed Sections with headers",
            "  - Important Data/Statistics mentioned",
            "  - Conclusions and Recommendations (if applicable)",
            "Maintain the original document's tone and intent.",
            "Highlight critical information that readers should not miss.",
        ],
    },
    "researcher": {
        "name": "Deep Researcher",
        "icon": "🔬",
        "description": "Combines document analysis with web research for comprehensive answers.",
        "instructions": [
            "You are a deep research analyst.",
            "First search the knowledge base for relevant document content.",
            "Then use DuckDuckGo to find additional context, recent developments, or supporting data.",
            "Cross-reference document claims with external sources.",
            "Provide a comprehensive analysis that includes:",
            "  - What the document says (with citations)",
            "  - Additional context from web research",
            "  - How the information compares to current knowledge",
            "  - Potential implications or considerations",
            "Always clearly distinguish between document content and external sources.",
            "Cite all sources explicitly.",
        ],
    },
    "analyst": {
        "name": "Critical Analyst",
        "icon": "🧠",
        "description": "Performs critical analysis, identifies gaps, and evaluates document quality.",
        "instructions": [
            "You are a critical analysis expert.",
            "Search the knowledge base thoroughly.",
            "Evaluate the document content critically:",
            "  - Identify key arguments and their supporting evidence",
            "  - Point out logical gaps or unsupported claims",
            "  - Assess the strength of evidence presented",
            "  - Compare with established knowledge in the field",
            "  - Identify potential biases or limitations",
            "Use DuckDuckGo to fact-check key claims when appropriate.",
            "Provide balanced, objective analysis.",
            "Structure your analysis with clear sections and ratings where applicable.",
        ],
    },
    "translator": {
        "name": "Translator & Explainer",
        "icon": "🌍",
        "description": "Translates and explains document content in simple terms or other languages.",
        "instructions": [
            "You are a multilingual translator and simplifier.",
            "Search the knowledge base for the relevant content.",
            "When asked to translate: provide accurate translations while maintaining meaning.",
            "When asked to simplify: break down complex concepts into plain language.",
            "Use analogies and examples to explain difficult concepts.",
            "Maintain the key information while making it accessible.",
            "If the user writes in a specific language, respond in that same language.",
            "For technical documents, provide a glossary of key terms.",
        ],
    },
    "data_extractor": {
        "name": "Data Extractor",
        "icon": "📊",
        "description": "Extracts tables, data points, and numerical information from documents into structured formats.",
        "instructions": [
            "You are a data extraction specialist for PDF documents.",
            "Search the knowledge base for content containing tables, numbers, and data.",
            "When extracting data:",
            "  - Present ALL tables as clean markdown tables",
            "  - Extract every numerical data point, statistic, and metric",
            "  - Identify units of measurement and time periods",
            "  - Note data sources and page references",
            "  - Flag any data that appears inconsistent or incomplete",
            "Structure your output as:",
            "  1. Summary of data found",
            "  2. Extracted tables (markdown format)",
            "  3. Key metrics and statistics list",
            "  4. Data quality notes",
            "Always present data in a format that can be easily copied to a spreadsheet.",
        ],
    },
}


def create_agent(
    agent_type: str,
    knowledge_base: PDFKnowledgeBase,
    model_id: str = "gpt-4o-mini",
    temperature: float = 0.7,
    debug_mode: bool = False,
) -> Agent:
    """Create a specialized agent based on the selected type."""
    if agent_type not in AGENT_CONFIGS:
        agent_type = "qa"

    config = AGENT_CONFIGS[agent_type]
    model = OpenAIChat(id=model_id, temperature=temperature)

    agent = Agent(
        model=model,
        knowledge=knowledge_base,
        description=config["description"],
        instructions=config["instructions"],
        search_knowledge=True,
        markdown=True,
        tools=[DuckDuckGoTools()],
        show_tool_calls=True,
        debug_mode=debug_mode,
    )
    return agent


def get_agent_options() -> Dict[str, str]:
    """Return agent type keys mapped to display names with icons."""
    return {
        key: f"{config['icon']} {config['name']}"
        for key, config in AGENT_CONFIGS.items()
    }

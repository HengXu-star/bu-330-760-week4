"""Math agent that solves questions using tools in a ReAct loop."""

import json
import time

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from calculator import calculate

load_dotenv()

# Configure your model below. Examples:
#   "google-gla:gemini-2.5-flash"    (needs GOOGLE_API_KEY)
#   "openai:gpt-4o-mini"             (needs OPENAI_API_KEY)
#   "anthropic:claude-sonnet-4-6"    (needs ANTHROPIC_API_KEY)
#   "groq:qwen/qwen3-32b"            (needs GROQ_API_KEY)
MODEL = "groq:qwen/qwen3-32b"

agent = Agent(
    MODEL,
    system_prompt=(
        "You are a helpful assistant. Solve each question step by step, but keep your reasoning and final answers concise. "
        "Prefer short answers and avoid long explanations unless needed. "
        "Use the calculator tool for every arithmetic or numeric calculation. "
        "Do not do math in your head or only in the text response. "
        "Use the product_lookup tool when a question mentions products from the catalog. "
        "For product questions, first call product_lookup for each product you need, then use the returned numeric prices directly in calculator expressions. "
        "Do not invent, assume, or rename prices into variables that the calculator has not been given. "
        "If you have the needed prices, compute the final numeric answer instead of saying the answer cannot be determined. "
        "If a question cannot be answered with the information given, say so."
    ),
)


@agent.tool_plain
def calculator_tool(expression: str) -> str:
    """Evaluate a math expression and return the result.

    Examples: "847 * 293", "10000 * (1.07 ** 5)", "23 % 4"
    """
    return calculate(expression)


@agent.tool_plain
def product_lookup(product_name: str) -> str:
    """Look up the price of a product by name.
    Use this when a question asks about product prices from the catalog.
    """
    with open("products.json") as f:
        products = json.load(f)

    if product_name in products:
        return str(products[product_name])

    return ", ".join(products.keys())


def load_questions(path: str = "math_questions.md") -> list[str]:
    """Load numbered questions from the markdown file."""
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and line[0].isdigit() and ". " in line[:4]:
                questions.append(line.split(". ", 1)[1])
    return questions


def main():
    questions = load_questions()
    for i, question in enumerate(questions, 1):
        print(f"## Question {i}")
        print(f"> {question}\n")

        for attempt in range(3):
            try:
                result = agent.run_sync(question)
                break
            except ModelHTTPError as e:
                if e.status_code == 429 and attempt < 2:
                    time.sleep(2)
                    continue
                raise

        print("### Trace")
        for message in result.all_messages():
            for part in message.parts:
                kind = part.part_kind
                if kind in ("user-prompt", "system-prompt"):
                    continue
                elif kind == "text":
                    print(f"- **Reason:** {part.content}")
                elif kind == "tool-call":
                    print(f"- **Act:** `{part.tool_name}({part.args})`")
                elif kind == "tool-return":
                    print(f"- **Result:** `{part.content}`")

        print(f"\n**Answer:** {result.output}\n")
        print("---\n")


if __name__ == "__main__":
    main()

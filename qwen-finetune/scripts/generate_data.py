"""
LLM-assisted training data generation.
Uses Claude or GPT-4 to generate examples matching your style.
"""

import json
import os
from pathlib import Path
import httpx

# Configure your API
API_KEY = os.environ.get("ANTHROPIC_API_KEY")  # or OPENAI_API_KEY
API_URL = "https://api.anthropic.com/v1/messages"  # or OpenAI endpoint


def load_style_guide(path: str = "data/raw/style_guide.md") -> str:
    """Load your style guide."""
    with open(path) as f:
        return f.read()


def load_seed_examples(path: str = "data/raw/seed_examples.jsonl", n: int = 5) -> list:
    """Load a few seed examples for few-shot prompting."""
    with open(path) as f:
        examples = [json.loads(line) for line in f if line.strip()]
    return examples[:n]


def generate_example(
    topic: str,
    style_guide: str,
    seed_examples: list,
    api_key: str = API_KEY,
) -> dict:
    """Generate a single training example using an LLM."""

    # Format seed examples for the prompt
    examples_text = ""
    for i, ex in enumerate(seed_examples, 1):
        convos = ex["conversations"]
        user_msg = next(c["content"] for c in convos if c["role"] == "user")
        asst_msg = next(c["content"] for c in convos if c["role"] == "assistant")
        examples_text += f"\nExample {i}:\nUser: {user_msg}\nAssistant: {asst_msg}\n"

    prompt = f"""Generate a training example for fine-tuning an LLM.

STYLE GUIDE:
{style_guide}

EXAMPLE OUTPUTS (match this style exactly):
{examples_text}

TOPIC FOR NEW EXAMPLE:
{topic}

Generate a realistic user question about this topic and an assistant response that perfectly matches the style guide. The response should demonstrate both the format/structure AND any domain-specific terminology from the style guide.

Output as JSON:
{{"conversations": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}

JSON only, no other text:"""

    # Call API (example for Anthropic)
    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }

    response = httpx.post(API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    content = response.json()["content"][0]["text"]

    # Parse JSON from response
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        import re
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse JSON from: {content}")


def generate_dataset(
    topics: list[str],
    output_path: str = "data/raw/generated.jsonl",
    style_guide_path: str = "data/raw/style_guide.md",
    seed_path: str = "data/raw/seed_examples.jsonl",
):
    """Generate a dataset from a list of topics."""

    style_guide = load_style_guide(style_guide_path)
    seed_examples = load_seed_examples(seed_path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    generated = []
    errors = []

    for i, topic in enumerate(topics):
        print(f"[{i+1}/{len(topics)}] Generating: {topic[:50]}...")
        try:
            example = generate_example(topic, style_guide, seed_examples)
            generated.append(example)

            # Save incrementally
            with open(output_path, "a") as f:
                f.write(json.dumps(example) + "\n")

        except Exception as e:
            print(f"  Error: {e}")
            errors.append({"topic": topic, "error": str(e)})

    print(f"\nGenerated {len(generated)} examples, {len(errors)} errors")
    print(f"Saved to {output_path}")

    if errors:
        error_path = output_path.replace(".jsonl", "_errors.json")
        with open(error_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"Errors saved to {error_path}")


# Example topic list for generation
EXAMPLE_TOPICS = [
    "What is Docker and why use it?",
    "How do microservices communicate?",
    "Explain database indexing",
    "What is CI/CD?",
    "How does caching work?",
    "What are environment variables?",
    "Explain API rate limiting",
    "What is a load balancer?",
    "How do webhooks work?",
    "What is OAuth?",
    # Add more topics for your domain...
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--topics-file", help="File with topics (one per line)")
    parser.add_argument("--output", default="data/raw/generated.jsonl")
    parser.add_argument("--style-guide", default="data/raw/style_guide.md")
    parser.add_argument("--seeds", default="data/raw/seed_examples.jsonl")
    args = parser.parse_args()

    if args.topics_file:
        with open(args.topics_file) as f:
            topics = [line.strip() for line in f if line.strip()]
    else:
        topics = EXAMPLE_TOPICS
        print("Using example topics. Create a topics file for your domain.")

    generate_dataset(
        topics=topics,
        output_path=args.output,
        style_guide_path=args.style_guide,
        seed_path=args.seeds,
    )

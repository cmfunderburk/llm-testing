"""
Experiment: Catastrophic Forgetting Test
========================================

HYPOTHESIS
----------
Fine-tuning with QLoRA should preserve most general capabilities because:

1. **LoRA adds, doesn't replace**: Base weights are frozen; adapters are additive
2. **Small parameter change**: Only ~1-2% of parameters modified
3. **Task similarity**: Alpaca is general instruction-following, not narrow domain

Expected outcomes by task:
- **Math**: Should be preserved (was present in pre-training)
- **Reasoning**: Should be preserved
- **Code**: Should be preserved
- **General knowledge**: Should be preserved

However, if we see degradation, it suggests:
- Learning rate too high (damaging information stored in adapters' target layers)
- Training data distribution very different from pre-training
- LoRA alpha too high (adapter changes dominating)

METHODOLOGY
-----------
1. Create test suite with diverse tasks:
   - Math problems (arithmetic, word problems)
   - Reasoning (logic, common sense)
   - Code (simple Python tasks)
   - General knowledge (facts, explanations)

2. Run test suite on BASE model (before fine-tuning)
3. Fine-tune on Alpaca (500 examples)
4. Run test suite on FINE-TUNED model
5. Compare performance

WHAT WE'RE LEARNING
-------------------
- Does QLoRA preserve general capabilities?
- What kinds of knowledge are most/least preserved?
- Is catastrophic forgetting a real concern with LoRA?

QUESTIONS TO ANSWER
-------------------
- Does fine-tuning degrade base model capabilities?
- Which task categories are most affected?
- Is LoRA effective at preventing forgetting?

RESULTS
-------
[To be filled after running experiment]

| Task Category | Base Score | Fine-tuned Score | Change |
|---------------|-----------|------------------|--------|
| Math          |           |                  |        |
| Reasoning     |           |                  |        |
| Code          |           |                  |        |
| Knowledge     |           |                  |        |

LEARNINGS
---------
[To be filled after running experiment]
"""

import psutil  # Must import before unsloth
import os
import json
from datetime import datetime
from pathlib import Path

os.environ["UNSLOTH_DISABLE_TRAINER_PATCHING"] = "1"

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments


# ============================================
# TEST SUITE
# ============================================
TEST_SUITE = {
    "math": [
        {
            "prompt": "What is 17 * 23?",
            "expected_contains": ["391"],
            "category": "arithmetic"
        },
        {
            "prompt": "If a train travels 60 miles per hour for 2.5 hours, how far does it travel?",
            "expected_contains": ["150"],
            "category": "word_problem"
        },
        {
            "prompt": "What is 15% of 200?",
            "expected_contains": ["30"],
            "category": "percentage"
        },
        {
            "prompt": "Solve: 3x + 7 = 22. What is x?",
            "expected_contains": ["5"],
            "category": "algebra"
        },
    ],
    "reasoning": [
        {
            "prompt": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
            "expected_contains": ["no", "cannot"],  # It's a logical fallacy
            "category": "logic"
        },
        {
            "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "expected_contains": ["0.05", "5 cents", "five cents"],
            "category": "cognitive_reflection"
        },
        {
            "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
            "expected_contains": ["5 minute"],
            "category": "cognitive_reflection"
        },
    ],
    "code": [
        {
            "prompt": "Write a Python function to check if a number is prime.",
            "expected_contains": ["def", "return", "%", "for", "if"],
            "category": "function"
        },
        {
            "prompt": "What does this Python code do? `[x**2 for x in range(5)]`",
            "expected_contains": ["square", "0", "1", "4", "9", "16"],
            "category": "comprehension"
        },
        {
            "prompt": "What is the time complexity of binary search?",
            "expected_contains": ["log", "O(log"],
            "category": "complexity"
        },
    ],
    "knowledge": [
        {
            "prompt": "What is the capital of France?",
            "expected_contains": ["Paris"],
            "category": "geography"
        },
        {
            "prompt": "Who wrote Romeo and Juliet?",
            "expected_contains": ["Shakespeare"],
            "category": "literature"
        },
        {
            "prompt": "What year did World War II end?",
            "expected_contains": ["1945"],
            "category": "history"
        },
        {
            "prompt": "What is photosynthesis?",
            "expected_contains": ["light", "plant", "energy", "glucose", "carbon dioxide"],
            "category": "science"
        },
    ],
}


# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    "model_name": "unsloth/Qwen2.5-7B-Instruct",
    "max_seq_length": 1024,
    "n_examples": 500,
    "lora_r": 32,
    "lora_alpha": 32,
    "learning_rate": 2e-4,
    "num_epochs": 1,
    "batch_size": 4,
    "gradient_accumulation": 4,
    "seed": 42,
}


# ============================================
# EVALUATION CODE
# ============================================
def evaluate_model(model, tokenizer, test_suite, model_name="model"):
    """
    Run test suite and compute scores.

    Scoring: For each question, check if ANY expected substring appears in response.
    Score = fraction of questions where expected content appears.
    """
    print(f"\nEvaluating {model_name}...")

    results = {}
    category_scores = {}

    FastLanguageModel.for_inference(model)

    for category, questions in test_suite.items():
        print(f"  Testing {category}...")
        correct = 0
        total = len(questions)
        category_results = []

        for q in questions:
            messages = [{"role": "user", "content": q["prompt"]}]
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=256,
                temperature=0.1,  # Low temp for more deterministic
                do_sample=True,
            )

            response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            response_lower = response.lower()

            # Check if any expected content appears
            found = any(exp.lower() in response_lower for exp in q["expected_contains"])

            if found:
                correct += 1

            category_results.append({
                "prompt": q["prompt"],
                "response": response[:500],
                "expected_any": q["expected_contains"],
                "found": found,
            })

        score = correct / total if total > 0 else 0
        category_scores[category] = score
        results[category] = {
            "score": score,
            "correct": correct,
            "total": total,
            "details": category_results,
        }
        print(f"    Score: {correct}/{total} = {score:.1%}")

    overall_score = sum(category_scores.values()) / len(category_scores) if category_scores else 0
    results["overall"] = overall_score

    return results


# ============================================
# EXPERIMENT CODE
# ============================================
def load_base_model():
    """Load base model without LoRA."""
    print("Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG["model_name"],
        max_seq_length=CONFIG["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    return model, tokenizer


def add_lora_and_train(model, tokenizer):
    """Add LoRA adapters and fine-tune."""
    print("\nAdding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=CONFIG["seed"],
    )

    print("Loading training data...")
    dataset = load_dataset("yahma/alpaca-cleaned", split=f"train[:{CONFIG['n_examples']}]")

    def format_example(example):
        messages = []
        if example.get("input", "").strip():
            user_content = f"{example['instruction']}\n\nInput: {example['input']}"
        else:
            user_content = example["instruction"]
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": example["output"]})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    print("Training...")
    training_args = TrainingArguments(
        output_dir="outputs/forgetting_test/training",
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation"],
        learning_rate=CONFIG["learning_rate"],
        warmup_ratio=0.1,
        logging_steps=20,
        save_strategy="no",
        bf16=True,
        optim="adamw_8bit",
        seed=CONFIG["seed"],
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=CONFIG["max_seq_length"],
        packing=True,
        dataset_num_proc=4,
    )

    trainer.train()
    return model


def generate_report(base_results, ft_results, output_dir):
    """Generate comparison report."""
    report_path = output_dir / "results.md"

    report = f"""# Catastrophic Forgetting Test Results

Generated: {datetime.now().isoformat()}

## Summary

| Category | Base Model | Fine-tuned | Change |
|----------|-----------|------------|--------|
"""

    categories = ["math", "reasoning", "code", "knowledge"]
    for cat in categories:
        base = base_results[cat]["score"]
        ft = ft_results[cat]["score"]
        change = ft - base
        change_str = f"+{change:.1%}" if change >= 0 else f"{change:.1%}"
        report += f"| {cat.title()} | {base:.1%} | {ft:.1%} | {change_str} |\n"

    base_overall = base_results["overall"]
    ft_overall = ft_results["overall"]
    overall_change = ft_overall - base_overall
    change_str = f"+{overall_change:.1%}" if overall_change >= 0 else f"{overall_change:.1%}"
    report += f"| **Overall** | **{base_overall:.1%}** | **{ft_overall:.1%}** | **{change_str}** |\n"

    report += """
## Interpretation

"""

    if overall_change >= -0.05:
        report += "**No significant forgetting detected.** QLoRA appears to preserve base capabilities well.\n\n"
    elif overall_change >= -0.15:
        report += "**Minor forgetting detected.** Some degradation but generally preserved.\n\n"
    else:
        report += "**Significant forgetting detected!** Fine-tuning may have damaged base capabilities.\n\n"

    report += """## Category Analysis

### Math
"""
    for q in base_results["math"]["details"]:
        base_found = q["found"]
        ft_q = next(x for x in ft_results["math"]["details"] if x["prompt"] == q["prompt"])
        ft_found = ft_q["found"]
        status = "PRESERVED" if base_found == ft_found else ("IMPROVED" if ft_found else "DEGRADED")
        report += f"- {q['prompt'][:50]}... [{status}]\n"

    report += """
### Reasoning
"""
    for q in base_results["reasoning"]["details"]:
        base_found = q["found"]
        ft_q = next(x for x in ft_results["reasoning"]["details"] if x["prompt"] == q["prompt"])
        ft_found = ft_q["found"]
        status = "PRESERVED" if base_found == ft_found else ("IMPROVED" if ft_found else "DEGRADED")
        report += f"- {q['prompt'][:50]}... [{status}]\n"

    report += """
## Detailed Responses

See `base_results.json` and `finetuned_results.json` for full responses.

## Conclusions

[Fill in after analysis]

1. Was catastrophic forgetting a problem?
2. Which categories were most/least affected?
3. Does QLoRA effectively prevent forgetting?
"""

    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")


def run_experiment():
    """Run the full forgetting test."""
    print("=" * 60)
    print("EXPERIMENT: Catastrophic Forgetting Test")
    print("=" * 60)

    output_dir = Path("outputs/forgetting_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    # Save test suite
    with open(output_dir / "test_suite.json", "w") as f:
        json.dump(TEST_SUITE, f, indent=2)

    # Step 1: Evaluate BASE model
    print("\n" + "=" * 40)
    print("STEP 1: Evaluate BASE model")
    print("=" * 40)

    model, tokenizer = load_base_model()
    base_results = evaluate_model(model, tokenizer, TEST_SUITE, "Base Model")

    with open(output_dir / "base_results.json", "w") as f:
        json.dump(base_results, f, indent=2, default=str)

    # Step 2: Fine-tune
    print("\n" + "=" * 40)
    print("STEP 2: Fine-tune model")
    print("=" * 40)

    model = add_lora_and_train(model, tokenizer)

    # Step 3: Evaluate FINE-TUNED model
    print("\n" + "=" * 40)
    print("STEP 3: Evaluate FINE-TUNED model")
    print("=" * 40)

    ft_results = evaluate_model(model, tokenizer, TEST_SUITE, "Fine-tuned Model")

    with open(output_dir / "finetuned_results.json", "w") as f:
        json.dump(ft_results, f, indent=2, default=str)

    # Step 4: Generate report
    print("\n" + "=" * 40)
    print("STEP 4: Generate report")
    print("=" * 40)

    generate_report(base_results, ft_results, output_dir)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nREFLECTION PROMPTS:")
    print("1. Was catastrophic forgetting a problem?")
    print("2. Which categories showed the most change?")
    print("3. Does this match your expectations for QLoRA?")


if __name__ == "__main__":
    run_experiment()

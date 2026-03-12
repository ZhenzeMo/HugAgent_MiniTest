# HugToM-QA Benchmark

Theory of Mind (ToM) evaluation benchmark. Given a person's demographic background and conversation history, test whether LLMs can correctly infer:

- **Belief Attribution** — What does this person believe causes what? (多选题, A/B)
- **Belief Update** — How would this person rate a follow-up survey question? (数值预测)

---

## Project Structure

```
hugagent_colm/
├── .env                   # API keys (do not commit)
├── .env.sample            # API key template
├── requirements.txt       # Python dependencies
└── Benchmark/
    ├── llm_utils.py       # All LLM client classes (QwenLLM, GeminiLLM, ChatGPTLLM, ...)
    ├── evaluate_qwen_5Q.py  # Main evaluation script (supports all models + max-context-qa)
    └── 54UsersQ/          # Benchmark datasets (6 JSONL files)
        ├── sample_belief_attribution_healthcare_filtered_different_results.jsonl   (108 questions)
        ├── sample_belief_attribution_surveillance_filtered_different_results.jsonl (122 questions)
        ├── sample_belief_attribution_zoning_filtered_different_results.jsonl       (126 questions)
        ├── sample_belief_update_healthcare_filtered_context_improves.jsonl         (472 questions)
        ├── sample_belief_update_surveillance_filtered_context_improves.jsonl       (364 questions)
        └── sample_belief_update_zoning_filtered_context_improves.jsonl             (550 questions)
```

---

## Step 1: Environment Setup

### 1.1 Create and activate conda environment

```bash
conda create -n hugtom python=3.9 -y
conda activate hugtom
```

### 1.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 1.3 Configure API keys

```bash
# Copy the template
cp .env.sample .env

# Edit .env and fill in your keys
# At minimum you need ONE key depending on which model you want to run:
#   Qwen / DeepSeek models  →  QWEN_API_KEY       (DashScope: https://dashscope.aliyuncs.com/)
#   Gemini models           →  GEMINI_API_KEY      (Google AI Studio: https://aistudio.google.com/)
#   GPT / o1 / o3 models   →  OPENAI_API_KEY      (OpenAI: https://platform.openai.com/)
#   Llama models            →  LLAMA_API_KEY       (Novita: https://novita.ai/)
#   Anthropic Claude        →  ANTHROPIC_API_KEY   (Console: https://console.anthropic.com/)
```

The `.env` file is automatically picked up by the scripts — no need to export variables manually.

---

## Step 2: Run Evaluation

All evaluation is done via `evaluate_qwen_5Q.py` (despite the name, it supports all models).

### Quickstart — single command

```bash
cd Benchmark

python evaluate_qwen_5Q.py \
  --benchmark_path 54UsersQ/sample_belief_attribution_healthcare_filtered_different_results.jsonl \
  --model qwen-plus
```

Results are saved automatically as a JSON file in the current directory:
```
qwen-plus-sample_belief_attribution_healthcare_filtered_different_results-20260311_120000.json
```

---

## Full Parameter Reference

| Parameter | Default | Description |
|---|---|---|
| `--benchmark_path` | `sample_prompt_v3.jsonl` | Path to the `.jsonl` benchmark file |
| `--model` | `qwen-plus` | Model to use (see supported models below) |
| `--temperature` | `0.1` | Generation temperature (0 = deterministic) |
| `--max-context-qa` | `None` (use all) | Limit context conversation turns fed to the model |
| `--max-workers` | `20` | Number of parallel API threads |
| `--output-path` | `.` (current dir) | Directory to save result JSON files |
| `--no-demographics` | off | Exclude participant demographics from prompt |
| `--no-context` | off | Exclude conversation history from prompt |
| `--swap-experiment` | off | Cross-participant sanity check (swap demographics/context between users) |
| `--debug` | off | Sequential mode: print full prompt + response for each question |

### Supported models

| Model name (pass to `--model`) | API Key needed |
|---|---|
| `qwen-plus` *(default)* | `QWEN_API_KEY` |
| `qwen-max` | `QWEN_API_KEY` |
| `qwen-turbo` | `QWEN_API_KEY` |
| `qwen2.5-72b-instruct` / `32b` / `14b` / `7b` / `3b` / `1.5b` / `0.5b` | `QWEN_API_KEY` |
| `deepseek-r1-0528` | `QWEN_API_KEY` (via DashScope) |
| `gemini-2.0-flash` | `GEMINI_API_KEY` |
| `gemini-1.5-pro` / `gemini-1.5-flash` / `gemini-2.5-flash-lite` | `GEMINI_API_KEY` |
| `gpt-4o` / `gpt-4o-mini` | `OPENAI_API_KEY` |
| `o1` / `o1-mini` / `o3-mini` | `OPENAI_API_KEY` |
| `gpt-5` / `gpt-5-mini` / `gpt-5.1` | `OPENAI_API_KEY` |
| `meta-llama/llama-3.3-70b-instruct` | `LLAMA_API_KEY` |

---

## Step 3: Common Run Examples

### Run all 6 benchmark datasets with qwen-plus

```bash
cd Benchmark

for FILE in 54UsersQ/*.jsonl; do
  python evaluate_qwen_5Q.py \
    --benchmark_path "$FILE" \
    --model qwen-plus \
    --output-path results/
done
```

### Control context length (the "5Q" feature)

```bash
# Use only the first 5 conversation turns as context
python evaluate_qwen_5Q.py \
  --benchmark_path 54UsersQ/sample_belief_attribution_healthcare_filtered_different_results.jsonl \
  --model qwen-plus \
  --max-context-qa 5

# Use first 10 turns
python evaluate_qwen_5Q.py \
  --benchmark_path 54UsersQ/sample_belief_update_surveillance_filtered_context_improves.jsonl \
  --model qwen-plus \
  --max-context-qa 10
```

### Run with a different model (e.g., GPT-4o)

```bash
python evaluate_qwen_5Q.py \
  --benchmark_path 54UsersQ/sample_belief_attribution_zoning_filtered_different_results.jsonl \
  --model gpt-4o \
  --temperature 0 \
  --output-path results/
```

### Ablation: no demographics, no context

```bash
# No demographics
python evaluate_qwen_5Q.py \
  --benchmark_path 54UsersQ/sample_belief_attribution_healthcare_filtered_different_results.jsonl \
  --model qwen-plus \
  --no-demographics

# No context (demographics only)
python evaluate_qwen_5Q.py \
  --benchmark_path 54UsersQ/sample_belief_attribution_healthcare_filtered_different_results.jsonl \
  --model qwen-plus \
  --no-context

# Neither (random baseline)
python evaluate_qwen_5Q.py \
  --benchmark_path 54UsersQ/sample_belief_attribution_healthcare_filtered_different_results.jsonl \
  --model qwen-plus \
  --no-demographics \
  --no-context
```

### Debug mode (inspect each prompt + response interactively)

```bash
python evaluate_qwen_5Q.py \
  --benchmark_path 54UsersQ/sample_belief_attribution_healthcare_filtered_different_results.jsonl \
  --model qwen-plus \
  --debug
```

In debug mode the script runs sequentially and shows the full constructed prompt, raw response, extracted answer, and whether it is correct. It pauses between questions and waits for `Enter`.

### Cross-participant swap experiment (sanity check)

```bash
python evaluate_qwen_5Q.py \
  --benchmark_path 54UsersQ/sample_belief_attribution_healthcare_filtered_different_results.jsonl \
  --model qwen-plus \
  --swap-experiment
```

Each participant's demographics and context are replaced with the **next** participant's data. If accuracy drops significantly vs. the normal run, it confirms the model is actually leveraging the provided context rather than guessing.

---

## Step 4: Understanding the Output

### Console output

```
Evaluating 108 questions across 2 groups...
Task types found: belief_attribution
============================================================
EVALUATING DIFFICULTY LEVEL: LONG
============================================================
Questions in this group: 54
Processing long: 100%|████████| 54/54 [01:23<00:00]

[LONG] Question 1:
Task Type: belief_attribution
Context QAs: 23
Task: Based on this conversation, what does the person believe about...
Generated: A  (Response: {"answer": "A"})
Correct: A
Method Used: function_call
Result: ✓
--------------------------------------------------
...
==================================================
RESULTS FOR LONG DIFFICULTY
==================================================
Total questions: 54
Correct answers: 38
Accuracy: 70.37%
Context QAs per question: 23

============================================================
SUMMARY RESULTS ACROSS ALL DIFFICULTIES
============================================================
Overall accuracy: 68.52% (74/108)

Breakdown by group:
  Long         (23 context): 70.37% (38/54)
  Short         (5 context): 66.67% (36/54)
```

### Result JSON schema

```json
{
  "timestamp": "2026-03-11 12:00:00",
  "model": "qwen-plus",
  "benchmark_path": "54UsersQ/sample_belief_attribution_healthcare_filtered_different_results.jsonl",
  "temperature": 0.1,
  "include_demographics": true,
  "include_context": true,
  "swap_experiment": false,
  "max_context_qa": null,
  "overall_metrics": {
    "total_questions": 108,
    "correct_answers": 74,
    "overall_accuracy": 0.685,
    "overall_mae": null,
    "overall_mse": null
  },
  "group_results": {
    "long": { "total": 54, "correct": 38, "accuracy": 0.703, ... },
    "short": { "total": 54, "correct": 36, "accuracy": 0.667, ... }
  },
  "all_question_details": [
    {
      "question_anchor": "id_prolificID",
      "question_id": "...",
      "prolific_id": "...",
      "vqa": { "task_question": "...", "answer_options": {...}, "answer": "A", ... },
      "generated_answer": "A",
      "generated_response": "{\"answer\": \"A\"}",
      "correct_answer": "A",
      "is_correct": true,
      "method_used": "function_call"
    }
  ]
}
```

- **`belief_attribution`** tasks: `overall_mae` / `overall_mse` are `null`; correctness is exact match (A/B)
- **`belief_update`** tasks: `overall_mae` / `overall_mse` are filled; correctness = within ±1 on 1–5 scale, ±2 on 1–10 scale

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'llm_utils'`

Run the script **from inside the `Benchmark/` directory**:

```bash
cd Benchmark
python evaluate_qwen_5Q.py ...
```

### `ValueError: DashScope API key is required`

Make sure `.env` exists in the project root (`hugagent_colm/`) and contains `QWEN_API_KEY=sk-...`.

### High invalid response rate (>5%) warning

The script automatically saves an error log `high_error_rate_*.json`. This usually means API rate limits were hit. Try:

```bash
# Reduce parallel workers
python evaluate_qwen_5Q.py ... --max-workers 5
```

### Crash log saved

If the script crashes entirely, a `crash_log_*.json` is saved to `--output-path`. Check the `error` and `original_error` fields for the root cause.

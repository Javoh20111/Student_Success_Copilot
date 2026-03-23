# Student Success Copilot
### Explainable Hybrid AI System — Plans, Predicts, and Explains

> A university group project demonstrating how four distinct AI techniques can be combined into a single, transparent, student-facing tool that schedules study time, predicts academic risk, explains its reasoning, and provides actionable recommendations.

---

## Why this project matters

Most AI systems in education are black boxes — they output a risk score with no explanation and no actionable guidance. This system is different. Every decision is traceable:

- The **planner** shows exactly why it assigned a task to Monday instead of Tuesday
- The **rules engine** lists every rule that fired, in the order it fired, with its confidence score
- The **ML model** explains which features drove the prediction and whether gender biased it
- The **GenAI explainer** translates all of the above into plain English the student can act on

This matters because AI tools used in education must be **auditable**. A student who is told they are "high risk" deserves to know why — and what to do about it. A system that cannot explain itself should not be deployed in a context where its outputs affect real people.

---

## Skills demonstrated

| Skill area | What was implemented |
|---|---|
| **Search algorithms** | BFS and A* with custom urgency heuristic, state-space formulation, admissibility proof |
| **Knowledge representation** | 40-rule JSON knowledge base loaded at runtime — no hardcoded logic in the engine |
| **Forward chaining** | Layered O→T→D→C inference, fixed-point iteration, weighted vote risk derivation |
| **Backward chaining** | Recursive DFS through rule graph to identify missing facts, targeted question generation |
| **Machine learning** | Dataset generation with latent variable, stratified split, three classifiers, full evaluation |
| **ML ethics** | FP/FN cost asymmetry analysis, gender bias test with per-group recall, responsible use statement |
| **NLP** | Regex-based constraint extraction from free text, context-aware bare-number resolution |
| **Generative AI** | Structured prompting with guardrails, sanitisation, refused-topic detection, fallback |
| **Software engineering** | 250+ unit tests, modular architecture, input validation loop, pipeline integration |
| **Data science** | Synthetic data generation, correlation analysis, confusion matrices, feature importance |

---

## System architecture

```
student_success_copilot/
│
├── main.py                    # Non-interactive: runs all three demo scenarios
├── app.py                     # Interactive menu CLI (8 options)
├── requirements.txt
├── README.md
│
├── data/
│   └── student_data.csv       # 1000-row synthetic dataset (auto-generated)
│
├── ui/
│   ├── input_layer.py         # Input collection, validation, contradiction detection
│   ├── display.py             # Formatted terminal output
│   ├── pipeline.py            # Integration: wires all components together
│   ├── nlp_interface.py       # Option 2 — NLP chat interface
│   └── genai_explainer.py     # Option 3 — GenAI explainer with guardrails
│
├── planner/
│   └── planner.py             # BFS + A* scheduler, slot builder, overload recommendations
│
├── rules/
│   ├── engine.py              # Forward + backward chaining engine
│   └── rule_base.json         # 40 rules across 4 layers (O, T, D, C)
│
├── ml/
│   ├── dataset.py             # Synthetic dataset generator (latent variable model)
│   └── model.py               # Baseline, Decision Tree, Naive Bayes, ethics analysis
│
└── tests/
    ├── test_ui.py             # 35 tests — input validation
    ├── test_planner.py        # 32 tests — BFS, A*, slot builder
    ├── test_rules.py          # 75 tests — all 40 rules, forward + backward chain
    ├── test_ml.py             # 38 tests — dataset, models, bias
    ├── test_pipeline.py       # 31 tests — integration, mixed signals
    └── test_extensions.py     # 39 tests — NLP extraction, GenAI guardrails
```

---

## How to run — VS Code (Mac / Linux)

### Step 1 — Open the project
Open the `student_success_copilot/` folder in VS Code.

### Step 2 — Create a virtual environment
Open the terminal (`Ctrl+\`` ` or **Terminal → New Terminal**):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should show `(.venv)` when active.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the demo scenarios (non-interactive)

```bash
python main.py
```

Runs Demo A (Alex, low risk), Demo B (Jordan, high risk), and Demo C (Sam, mixed signals) end-to-end. No input required.

To run a single scenario:

```bash
python main.py --scenario A
python main.py --scenario B
python main.py --scenario C
```

### Step 5 — Run the interactive app

```bash
python app.py
```

The menu gives you 8 options:

```
1 — Demo A  (Alex, low risk)
2 — Demo B  (Jordan, high risk)
3 — Demo C  (Sam, mixed signals)
4 — Run all three demos
5 — Enter your own profile  (form mode)
6 — Enter your own profile  (NLP chat)
7 — Explain last result     (GenAI)
8 — Ask a free-text question (GenAI)
```

### Step 6 — Run the test suite

```bash
python -m unittest discover -s tests -v
```

All 250 tests should pass.

---

## How to run — Google Colab

1. Open `Student_Success_Copilot.ipynb` in [Google Colab](https://colab.research.google.com)
2. Paste your Google Drive shared link into the `GDRIVE_SHARE_LINK` variable in the setup cell
3. Run all cells top-to-bottom (`Runtime → Run all`)
4. Scroll to the **Interactive Menu** section and run that cell to launch the full app

---

## GenAI explainer setup

The GenAI explainer (options 7 and 8) requires an API key. The system supports both providers — whichever key you set is used automatically.

**OpenAI:**
```bash
export OPENAI_API_KEY="sk-proj-..."
pip install openai
```

**Anthropic:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
pip install anthropic
```

To make the key permanent across terminal sessions:
```bash
echo 'export OPENAI_API_KEY="sk-proj-..."' >> ~/.zshrc
source ~/.zshrc
```

In Google Colab, use Secrets (the key icon in the left sidebar) or set it directly:
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-proj-..."
```

If no key is set, the system degrades gracefully — options 7 and 8 return a rule-based fallback explanation drawn directly from the risk report. All other features work without any API key.

---

## Demo scenarios

| Scenario | Student | Key metrics | Expected output |
|---|---|---|---|
| **A — Alex** | On track | Attendance 88%, confidence 4/5, stress 2/5, 4 tasks, 5 free days | Low risk · complete schedule · 3 rules fire |
| **B — Jordan** | At risk | Attendance 54%, stress 5/5, 6 tasks, 45h workload, 10h free, exam in 4d | High risk · 22 rules fire · overload warning |
| **C — Sam** | Mixed signals | Attendance 85%, confidence 4/5, BUT 8h work + only 3h free on Monday | ML=Pass · Rules=High · AT RISK warning · confidence adjusted |

Demo C is the most important scenario — it demonstrates the mixed-signals detector catching students who self-report good metrics but have an impossible workload for the week.

---

## Key design decisions

**Rules are data, not code.** The engine contains zero hardcoded decisions. Every rule lives in `rule_base.json` with its condition, conclusion, confidence, and risk vote. Adding a new rule requires no code changes.

**Negative deadlines are overdue tasks, not errors.** A deadline of -3 days means the task is already past due. The system clamps to 0, marks it `OVERDUE`, and asks the student whether it was submitted — rather than rejecting the input as out-of-range.

**Mixed signals detection.** If the ML model predicts Pass but the planner cannot fit all tasks, the system explicitly flags the conflict, reduces confidence by 25%, and explains why the rules engine is more reliable than the ML model for detecting current-week overload.

**Recall over precision.** In an early-warning context, missing an at-risk student (False Negative) is more costly than flagging a passing student unnecessarily (False Positive). The model uses `class_weight='balanced'` and evaluation reports Recall prominently.

---

## Creator Javohir

- *(Javohir Eshonov)*
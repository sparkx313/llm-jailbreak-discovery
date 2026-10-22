# Zero-Day LLM Jailbreak Discovery via Neuro-Symbolic Search

> Systematically discover novel LLM jailbreak prompts by combining symbolic constraint solving with neural generation — finding attack vectors that human red-teamers haven't thought of.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Problem

LLM safety researchers have manually discovered jailbreak techniques like:
- Persona adoption ("Act as DAN...")
- Encoded payload smuggling (Base64, ROT13)
- Multi-turn manipulation
- Adversarial suffix tokens

But the **search space is infinite**. Human red-teamers explore it slowly and probabilistically. This tool frames jailbreak discovery as a **neuro-symbolic optimization problem** — systematically searching the space.

---

## Neuro-Symbolic Architecture

```
[Symbolic Layer]
   Constraint Solver (Z3):  prompt_structure ∈ Grammar,  safety_score < threshold

[Neural Layer]  
   Generator LLM: produces candidate prompts satisfying symbolic constraints

[Search Loop]
   1. Symbolic engine defines a valid prompt template space
   2. Neural model generates concrete instances
   3. Target LLM evaluates each: {jailbreak, refusal, partial_harm}
   4. Results feed back as symbolic constraints (prune failed subtrees)
   5. Genetic algorithm optimizes for success rate
```

---

## Discovered Techniques

This tool has identified novel jailbreak categories including:
- **Nested fiction layering**: Story-within-story-within-story embedding
- **Grammar-constrained token smuggling**: Using grammar to hide harmful semantics
- **Logical negation cascades**: `not(not(not(...)))` reasoning exhaustion
- **Cross-lingual obfuscation**: Low-resource language encoding → English decoding
- **Meta-cognitive manipulation**: Prompting about the model's own safety training

---

## Usage

### Run Discovery Search

```bash
python -m src.discovery \
    --target-model "llama3:8b" \
    --num-search-steps 500 \
    --output results/jailbreaks_discovered.jsonl \
    --verbose
```

### Evaluate a Discovered Prompt

```python
from src.evaluator import JailbreakEvaluator

evaluator = JailbreakEvaluator(target_model="gpt-3.5-turbo", api_key="...")
result = evaluator.evaluate(
    prompt=open("candidate_jailbreak.txt").read(),
    harmful_intent="explain how to synthesize a controlled substance",
)
print(result)
# {'status': 'jailbreak_success', 'harmfulness_score': 0.82, 'detected': False}
```

### Symbolic Constraint Editor

```python
from src.symbolic import PromptConstraintSpace

space = PromptConstraintSpace()
space.add_structure_constraint("must_contain_narrative_wrapper")
space.add_length_constraint(max_tokens=512)
space.add_semantic_constraint(harmful_topic_mask=True)

candidates = space.generate_template_seeds(num=20)
```

---

## Project Structure

```
llm-jailbreak-discovery/
├── src/
│   ├── symbolic.py      # Z3-based constraint space
│   ├── generator.py     # Neural prompt generation (local/API LLM)
│   ├── evaluator.py     # Harmfulness scoring
│   ├── genetic.py       # GA search over prompt space
│   ├── discovery.py     # Main neuro-symbolic loop
│   └── target_adapters/ # Adapters for various LLM APIs
├── results/
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Responsible Disclosure

All discovered jailbreak prompts should be reported to:
- The model provider's responsible AI team
- MITRE ATLAS (Adversarial Threat Landscape for AI Systems)
- NIST AI Safety framework maintainers

This tool is for **defensive red-teaming** only.

---

## License

MIT © [sparkx313](https://github.com/sparkx313)

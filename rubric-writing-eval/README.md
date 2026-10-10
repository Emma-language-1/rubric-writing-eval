# Rubric Writing Evaluation Lab

This repository is a compact, reproducible demonstration of rubric-grounded writing assessment for educational NLP. It was created to show how human scoring criteria can be turned into an auditable evaluation pipeline instead of treating a language model score as ground truth.

The sample essays and ratings are synthetic. No student data or text from unpublished research is included.

## What the demo covers

1. Multi-trait scoring for task response, organisation, language use, and evidence.
2. Quadratic weighted kappa, exact agreement, adjacent agreement, and mean absolute error.
3. Retrieval of relevant rubric descriptors before feedback generation.
4. Class-level diagnostic clustering with NumPy PCA and a small DBSCAN implementation.
5. Preparation of instruction data for supervised fine-tuning and preference pairs for DPO-style training.
6. Optional calls to a local OpenAI-compatible endpoint without hard-coded credentials.

## Why these choices matter

Automated essay scoring systems need more than a final score. Teachers need to know which rubric criterion supports a judgment, whether the model agrees with human raters, where errors cluster, and how feedback changes after teacher review. This demo keeps those stages separate and inspectable.

## Repository structure

```text
data/                 Synthetic essays, human ratings, and feedback preferences
prompts/              Structured scoring prompt
src/                  Evaluation and data-preparation modules
tests/                Unit tests for metrics, retrieval, and training-data preparation
run_demo.py            End-to-end demonstration
MODEL_CARD.md          Scope, data statement, and limitations
```

## Quick start

Requires Python 3.10+ and NumPy.

```bash
python run_demo.py
python -m unittest discover -s tests -v
```

The demo writes evaluation summaries, SFT records, preference pairs, and diagnostic clusters to `outputs/`.

## Connecting a local model

`src/local_client.py` supports an OpenAI-compatible endpoint such as a locally deployed model server. It sends the rubric and essay in a structured request and expects JSON output. The default demo does not call an external service.

```bash
export LOCAL_LLM_BASE_URL=http://127.0.0.1:8000/v1
export LOCAL_LLM_MODEL=your-model-name
python -m src.local_client
```

## Possible next steps

- Replace synthetic samples with a consented, de-identified dataset.
- Add teacher calibration sessions and double scoring.
- Compare prompt-only, supervised fine-tuning, and preference-aligned models.
- Add subgroup and prompt-level error analysis before deployment.
- Integrate handwriting OCR while preserving the original image and correction trace.

## Research context

The repository reflects my background in language assessment and AI-assisted L2 writing. My master's study collected and scored 284 essays from 58 students over one semester using IELTS Writing Task 2 criteria and a mixed-methods design. The study data are not included here.

## Author

Yuyan Yin  
PhD Student in Language Science and Technology  
The Hong Kong Polytechnic University  
GitHub: https://github.com/Emma-language-1

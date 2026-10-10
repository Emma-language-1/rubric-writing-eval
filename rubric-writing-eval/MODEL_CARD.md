# Model and evaluation card

## Intended use

This repository demonstrates components of a teacher-in-the-loop writing evaluation workflow. It is suitable for portfolio review, methods discussion, and low-risk experimentation.

## Data

All essays, labels, and preference pairs in `data/` are synthetic. They were written for demonstration and do not represent real students. The files must not be used to claim model accuracy or educational effectiveness.

## Scoring approach

The default scorer is a transparent heuristic baseline. It converts observable text features into four trait scores. It is included to test the evaluation pipeline, not as a production scoring model.

## Evaluation

The pipeline reports quadratic weighted kappa, exact agreement, adjacent agreement, and mean absolute error. These measures should be calculated separately by prompt, year level, language background, and other relevant subgroups when a consented dataset is available.

## Human oversight

Teachers should retain final authority. Model scores and feedback should be presented with rubric evidence and an editable audit trail.

## Limitations

- Synthetic data cannot establish generalisation.
- Surface features do not capture factual accuracy, originality, or deep reasoning.
- Chinese tokenisation in the baseline is character based.
- The demonstration does not train a foundation model.
- Local model responses require separate validity, bias, privacy, and security review.

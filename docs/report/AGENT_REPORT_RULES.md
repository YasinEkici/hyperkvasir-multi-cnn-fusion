# Agent Report Writing Rules

This document defines the writing rules for the final report of the HyperKvasir multi-CNN fusion project. The report has to be in Turkish.

The purpose is to make the report academically honest, project-specific, traceable to repository evidence, and clearly based on the actual engineering process followed in this project.

## 1. Core Writing Principle

The report must describe this specific project, not a generic deep learning study.

Avoid generic textbook-style openings such as:

> Deep learning has revolutionized medical image analysis in recent years.

Prefer project-specific writing such as:

> In this project, the main difficulty was not only achieving high accuracy, but also evaluating model behavior under an imbalanced 23-class protocol where rare classes can be hidden by aggregate metrics.

The report should sound like it was written from the actual experiment history, project decisions, limitations, and documented results.

## 2. Evidence-Based Writing

Every major claim must be traceable to one of the following:

* documented project decisions,
* experiment logs,
* result tables,
* result figures,
* documented configuration files,
* literature files already included in the repository.

Do not invent:

* numerical results,
* completed experiments,
* model improvements,
* comparison values,
* citations,
* runtime values,
* hardware details,
* conclusions that are not supported by results.

If a result is missing, write `TODO` instead of guessing.

## 3. Project-Specific Details to Include

The report should include concrete details from the actual implementation when relevant:

* HyperKvasir 23-class classification task,
* official 5-fold evaluation protocol,
* ResNet50, MobileNetV2, EfficientNetB0 backbones,
* feature extraction and projection dimension,
* concat fusion and weighted fusion,
* MLP-only classifier decision,
* macro-F1 as the headline metric,
* accuracy as a supporting metric,
* confidence intervals if documented,
* per-class analysis,
* confusion matrix interpretation,
* rare-class limitations,
* experiment IDs when useful,
* Colab A100 / provenance gate only if documented and relevant.

Do not overload the report with implementation trivia. Use details only when they support the methodology, reproducibility, or discussion.

## 4. Explain Decisions, Not Only Actions

The report should not only say what was done. It should explain why it was done.

Important questions to address:

* Why is macro-F1 the headline metric?
* Why is accuracy not enough for this dataset?
* Why is the classifier fixed as MLP?
* Why is the official 5-fold protocol used?
* Why are some literature results only contextual and not direct comparisons?
* Why should negative or neutral ablation results still be reported?
* Why is fusion expected to help, and when might it fail?

These explanations make the report more defensible and more connected to the actual engineering process.

## 5. Handling Negative or Neutral Results

Do not hide weak, negative, or neutral results.

If an ablation does not improve the previous best result, state it honestly.

Example:

> Although weighted fusion produced the strongest overall result, the improvement over concat fusion should be interpreted carefully because the confidence intervals are close.

This type of discussion is preferred over exaggerating small differences.

## 6. Literature Use and Citation Rules

Do not copy wording from papers.

When using literature:

* paraphrase in our own words,
* cite the source properly,
* separate what the paper reports from what our project concludes,
* avoid direct head-to-head claims if the protocols differ.

Example structure:

> Borgli et al. report both macro and micro metrics for HyperKvasir, which is important because class imbalance can make aggregate accuracy overly optimistic. In this report, this motivates using macro-F1 as the primary metric rather than treating accuracy as the only performance indicator.

External results must be described as contextual when:

* the split protocol differs,
* augmentation strategy differs,
* class balancing differs,
* dataset version differs,
* number of classes differs,
* training or evaluation setup is not directly comparable.

## 7. Claims That Must Not Be Made

Do not claim:

* state-of-the-art performance,
* direct superiority over papers with different protocols,
* completed focal/TTA/ensemble/recipe-v2 results unless documented,
* that accuracy alone proves success,
* that the model is clinically usable,
* that rare classes are solved if per-class metrics show weakness,
* that protocol-mismatched literature values are direct baselines.

Prefer careful wording:

* “competitive under the selected official protocol”
* “controlled comparison”
* “methodologically consistent evaluation”
* “contextual comparison with literature”
* “limited direct comparability”
* “the improvement should be interpreted cautiously”

## 8. Tone and Style

The writing should be:

* academic,
* clear,
* specific,
* honest,
* technically precise,
* connected to repository evidence.

Avoid:

* generic filler,
* marketing-style claims,
* exaggerated language,
* overly polished but empty sentences,
* repeated phrases,
* unnecessary buzzwords.

Do not write “make it human-like.” Instead, write in a project-specific and evidence-based way.

## 9. Manual Student Contribution

Some parts should be reviewed and manually refined by the student:

* final paragraph of the Introduction,
* explanation of why the protocol was selected,
* interpretation of the most important results,
* limitations,
* future work,
* final conclusion.

The final report should reflect the student’s understanding of the project, not only automatically generated text.

## 10. Section-Specific Rules

### Introduction

The Introduction should explain:

* the task,
* why the dataset is challenging,
* why feature fusion is investigated,
* what comparison axes the project studies.

Avoid broad generic claims unless they are directly relevant.

### Methodology

The Methodology should explain:

* backbones,
* feature extraction,
* projection,
* fusion methods,
* MLP classifier,
* frozen vs fine-tuning setup.

Do not introduce models or methods that were not implemented.

### Experimental Setup

The Experimental Setup should explain:

* dataset setup,
* official 5-fold protocol,
* metrics,
* hyperparameters if documented,
* reproducibility details if documented.

Do not fabricate runtime or hardware details.

### Results

The Results section must use only documented values.

It should present:

* single-backbone results,
* fusion results,
* concat vs weighted comparison,
* frozen vs fine-tuning comparison,
* confidence intervals if available,
* per-class results if available,
* confusion matrix if available.

### Discussion

The Discussion should explain what the results mean.

It should include:

* why macro-F1 matters,
* whether fusion helped,
* why some classes were difficult,
* why literature comparison is limited,
* what the project did well,
* what remains weak.

### Conclusion

The Conclusion should summarize the actual findings without exaggeration.

It should not introduce new results.

## 11. Final Integrity Checklist

Before finalizing the report, check:

* Are all numerical results traceable to repository files?
* Are all major claims supported?
* Are incomplete experiments clearly marked or omitted?
* Are literature comparisons fair?
* Is macro-F1 treated as the headline metric?
* Is accuracy not over-emphasized?
* Are negative or neutral results reported honestly?
* Are limitations included?
* Are there any generic paragraphs that could belong to any deep learning paper?
* Does the report reflect this specific project’s process?

If any answer is problematic, revise the report before final submission.

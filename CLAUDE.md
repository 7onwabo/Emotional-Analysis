4. Emotion Analysis using BRIGHTER + EthioEmo Datasets
Objective: Develop emotion classification models for African languages.

Dataset: BRIGHTER + EthioEmo, AfriSenti, AfriHate.

Key Tasks:

Implement multi-label emotion classification.
Use deep learning to detect nuanced emotions.
Compare model performance across African languages.
Your final report should follow a standard NLP research paper structure. Below are the required sections and brief guidelines on what to include. These  guidelines are designed to help you write clearly, demonstrate your understanding of the work, and incorporate responsible NLP and AI reflections.

Report Sections
1. Introduction
State the problem you are addressing.
Explain why it is important, especially for African or low-resource languages.
Briefly state your objectives and contributions.
Reflect briefly on the societal value or ethical motivation for your work.
Responsible NLP Reflection:

Why does your work matter? Are you addressing underrepresentation or fairness challenges?

2. Background
Summarise related work (datasets, models, previous research).
Define key concepts and terminology.
Identify gaps in prior work that your project aims to address.
Responsible NLP Reflection:

Highlight gaps in prior research for African languages and acknowledge possible limitations in prior datasets or methods.

3. Methodology
Describe your data sources and preprocessing steps.
Explain your model choices, training setup, and evaluation metrics.
Justify key decisions (e.g., model selection, hyperparameters).
Responsible NLP Reflection:

Where did your data come from?
How did you handle data limitations or potential biases?
Are there licensing or ethical considerations?
4. Experiments & Results
Present your quantitative results clearly (tables, figures, evaluation scores).
Include comparative baselines.
Provide qualitative error analysis if applicable.
Responsible NLP Reflection:

Did your models perform equally across languages?
Are there any observed fairness or bias issues?
What do your errors reveal about system limitations?
5. Reflections and Discussion
Discuss what worked, what didn’t, and lessons learned.
Describe any challenges you faced.
Suggest possible extensions or improvements.
Include a discussion on responsible AI principles:
Data openness & reproducibility
Fairness across languages
Limitations and risks
Broader societal impacts
6. Conclusion
Summarise your key findings and contributions.
Restate key lessons learned.
Mention any open questions or areas for future work.
📚 References
Use a consistent citation style (ACL recommended).
Include:
All papers, datasets, models, and code libraries used.
Any pretrained models, repositories, or open resources.
General Reminders
Clear writing is important: prioritize clarity over technical jargon.
Figures and tables should be easy to read and referenced in the text.
The report is part of your Content Mark for the project.
💡 Reflection on Responsible NLP
Throughout your report, demonstrate that you have considered:

Language diversity and resource availability.
Data access, openness, and licensing.
Equity, bias, and inclusivity.
The potential real-world impact of your system.
Even brief reflections on these issues show responsible AI thinking, which is important both academically and professionally.


Fit Page






Emotion Analysis for African Languages
Multilabel Emotion classification using BRIGHTER, EthioEmo, AfriSenti, and AfriHate
Members: Ibrahim Said, Lonwabo Kwitshana, Mpho Tsotetsi
Group Number: 43
Introduction
Emotion Analysis is the automated identification of emotional states extracted from a body of text. Despite
the rapid progress driven by large pre-trained models, the majority of emotion recognition systems cater
to well-known languages such as English, leaving other African languages severely underserved. This
proposal presents our plan to develop and evaluate multilabel emotion classification models for African
languages. This will be done by making use of four specialised corpora: BRIGHTER, EthioEmo, AfriSenti,
and AfriHate. This will address the gap in the literature by combining multi-label formulation, cross-lingual
comparison, and culturally grounded data to produce classifiers that can be practical for African NLP
communities.
Background and related work
Emotion recognition has roots in Plutchik's (1980) wheel of emotions, which organises affect into eight
primary categories and has shaped annotation schemes in BRIGHTER (Muhammad et al., 2025) and
EthioEmo (Belay et al., 2025). Early approaches relied on lexicon-based SVMs, which fail in low-resource
settings. SemEval-2018 Task 1 (Mohammad et al., 2018) standardized multilabel evaluation conventions,
including macro-F1 and Pearson r, while Demszky et al. (2020) showed transformers substantially
outperform feature-engineered baselines on GoEmotions. Both benchmarks, however, are English-only
and do not transfer to African contexts. Dominant cross-lingual models mBERT and XLM-RoBERTa
(Conneau et al., 2020) underrepresent African languages, motivating AfriBERTa (Ogueji et al., 2021) and
AfroXLMR (Alabi et al., 2022) Africa-specific models that achieve state-of-the-art results on African NLP
tasks and form the backbone of our experiments.
Proposed Methodology
Research Questions:
● RQ1: Can transformer models fine-tuned on BRIGHTER (Bianchi et al., 2024) and EthioEmo
(Yimam et al., 2024) achieve competitive multilabel emotion classification across African
languages?
● RQ2: How does model performance vary across African languages, and what linguistic factors
explain these gaps in performance?
● RQ3: Does supplementary data from AfriSenti (Muhammad et al., 2023) and AfriHate (Ayele et
al., 2023) improve classifier performance through transfer learning or data augmentation?

Datasets:
Dataset Languages Approximated Size Labels/ Task
BRIGHTER Multiple African + global ~30,000 instances Multi-label emotion
EthioEmo Amharic, Tigrinya, Oromo ~5,000 instances Multi-label emotion
AfriSenti 14 African languages ~110,000 tweets Sentiment (auxiliary)
AfriHate Multiple African languages ~13,000 instances Hate Speech (auxiliary)
Approach:
● Classical: TF-IDF features with a multilabel Logistic Regression classifier to establish a classic
NLP baseline.
● Deep Learning Baseline: BiLSTM with FastText embeddings trained on African language corpora.
● Transformer Fine-Tuning (Primary): AfriBERTa (Ogueji et al., 2021) and AfroXLMR (Alabi et al.,
2022) fine-tuned end-to-end with Binary Cross-Entropy loss and label-frequency weighting for
class imbalance.
● Cross-lingual Transfer: Zero-shot and few-shot evaluation of the best model on languages
withheld during training.
PHASE Work plan WEEKS
1)DATA Download, preprocess all datasets 1–2
2) BASELINES TF-IDF + LR and BiLSTM baselines 3–4
3)TRANSFORMER Fine-tune and AfroXLMR cross-lingual experiments 5–7
4) AUGMENT Back-translation and paraphrase augmentation 8–9
5) ANALYSIS Error analysis, per-language breakdowns, write-up 9-
Evaluation
All models will be evaluated using the following metrics, which were chosen to capture the multilabel
nature of the task:
● Macro-averaged F1-score - primary metric, treats each emotion class equally regardless of
frequency
● Micro-averaged Precision, Recall & F1 - aggregate over all label-instance pairs.
● Hamming Loss - fraction of incorrectly predicted labels per instance.
● Per-language F1 - enables direct cross-lingual comparison.
All experiments will be reproduced with fixed random seeds for reproducibility. A structured error analysis
will be conducted on misclassified instances to characterise systematic failure modes by language and
emotion category.

Expected Output
It is anticipated that transformer-based models will significantly outperform classical and BiLSTM
baselines on multilabel emotion classification across African languages. Cross-lingual transfer is expected
to show moderate zero-shot performance for typologically related language pairs, with performance
degrading for more distant pairs. The deliverables of this project are:
● Trained and evaluated multilabel emotion classifiers with reproducible code.
● Per-language and per-label performance benchmarks enabling future comparison.
● A structured error analysis which identifies linguistic and data-level challenges for African emotion
NLP.



some findings:
Found it — not a bug, and not a split problem. swa is intrinsically sparse:

  ┌─────┬───────────────┬────────────────┬────────────┬───────────────┬───────────────┐
  │     │ all-zero rows │ mean labels/ex │ fear prev. │ disgust prev. │ sadness prev. │
  ├─────┼───────────────┼────────────────┼────────────┼───────────────┼───────────────┤
  │ swa │ 44.5%         │ 0.60           │ 2.8%       │ 7.2%          │ 10.6%         │
  ├─────┼───────────────┼────────────────┼────────────┼───────────────┼───────────────┤
  │ hau │ 12.9%         │ 1.11           │ 15.6%      │ 15.3%         │ 30.4%         │
  └─────┴───────────────┴────────────────┴────────────┴───────────────┴───────────────┘

  Key facts:
  - swa train/test prevalence are near-identical (e.g. fear 0.028/0.028) → no distribution shift, no data leak, split is fine.
  - swa has 2–5× lower emotion prevalence than hau and 44% neutral (no-emotion) texts.
  - swa's most common label (joy, 13.5%) scores F1 0.52 — model predicts it fine. The rare labels collapse: fear (2.8%)→0, disgust (7.2%)→0.
  - swa recall_macro 0.158 ≪ precision 0.337 → systematic under-prediction of positives.

  Diagnosis: mechanism is class imbalance + a global 0.5 threshold + unweighted BCE. The model is trained on pooled multilingual data where hau/amh have far higher label density;
  at threshold 0.5 it rarely clears the bar for swa's rare labels, so their recall (and macro-F1) collapse. Macro-F1 punishes this hardest because it weights each rare class
  equally.

  This is two findings at once:
  - RQ2: a real linguistic/annotation factor — swa's BRIGHTER annotation is sparser and more neutral-heavy than hau/amh. That alone explains the gap; worth stating explicitly.
  - RQ1: it's exactly what proposal gap #3 (label-frequency weighting / pos_weight) + threshold tuning on dev are meant to fix. The low overall recall (0.48) is the same disease,
  swa just the worst case.
  implement pos_weight in the BCE loss + dev-tuned (per-label) thresholds, then re-eval — should lift swa rare-label recall and overall
  recall together.
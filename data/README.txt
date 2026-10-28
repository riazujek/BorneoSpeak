1. Unstructured Data Ingestion and LLM-Assisted Extraction
The pipeline begins with the digitization of Anthony Richards' An Iban-English Dictionary (1981). Because the source material was an OCR-scanned text document containing heavy noise, inconsistent formatting, and artifacts, traditional regex-based scraping was insufficient. A Generative AI extraction script was engineered to read the unstructured text in strategic chunks and map the linguistic data into structured key-value pairs (source_word, english_translation). To navigate API token windows and rate limits, the text was algorithmically sliced and processed in distinct batches, yielding the initial raw CSV datasets.

2. Dataset Sanitization and Master Dictionary Compilation
The raw LLM outputs required programmatic cleaning to become production-ready. A sanitization script was deployed to detect and strip mojibake (encoding errors resulting from the OCR process), remove duplicate entries, and standardize text formatting. Fragmented dataset files (e.g., Part 1 and Part 2) were merged into a single, pristine source of truth: cleaned_iban_dataset_MASTER.csv. This file serves as the definitive vocabulary baseline for both the active lookup engine and the machine learning model.

3. Synthetic Parallel Corpus Generation (Knowledge Distillation)
Because Neural Machine Translation (NMT) models require sentence-level context rather than isolated vocabulary, the pipeline utilizes a technique called Knowledge Distillation. A Python automation script feeds batches of the sanitized master dictionary into a cloud-based LLM. The script uses strict prompt engineering to anchor the LLM to the verified dictionary definitions, forcing it to generate grammatically accurate, real-world Iban-to-English sentence pairs while severely reducing hallucination.

To ensure the output could be fed directly into a neural network training loop, the prompt enforces strict JSON array formatting. The script includes automated error handling, progressive saving, and rate-limit pacing, resulting in the synthetic_parallel_corpus.csv—the exact training data required to fine-tune the MarianMT architecture.

4. Mobile-Optimized Database Compilation
The final stage of the data engineering pipeline transforms the data for edge deployment. The master CSV dictionary is compiled into a lightweight, relational SQLite database (.db). This allows the mobile application to query translations locally using SQL indexing, achieving zero-latency lookups entirely offline without relying on cloud infrastructure.


The Source Material (The Dictionary)

Richards, A. (1981). An Iban-English Dictionary. Oxford University Press. (Use this when introducing your base dataset and the challenges of digitizing indigenous texts).

The Methodology (Knowledge Distillation & Synthetic Data)

Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. arXiv preprint arXiv:1503.02531. (This is the foundational paper for using a massive "Teacher" model to train a smaller "Student" model).

Wang, Y., et al. (2021). Towards Zero-Label Language Learning. arXiv preprint arXiv:2109.09193. (A strong reference for using large language models to synthetically generate parallel corpora for low-resource languages).

Google. (2023). Gemini: A Family of Highly Capable Multimodal Models. Google DeepMind. (The official technical report to cite when discussing your use of the Gemini 3.1-flash-lite API).

The Architecture (MarianMT & Edge AI)

Junczys-Dowmunt, M., et al. (2018). Marian: Fast Neural Machine Translation in C++. Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics. (The official citation for the MarianMT framework you are using for your local, offline translation engine).

Lin, J., et al. (2020). MCUNet: Tiny Deep Learning on IoT Devices. Advances in Neural Information Processing Systems. (A great supporting reference for why running localized, zero-latency machine learning models on edge devices is necessary for rural deployment).

QA

To rigorously defend your Quality Assurance (QA) methodology—specifically the length-ratio heuristic, hallucination filtering, and manual sampling—you should include these academic references in your thesis:
**Heuristic Corpus Filtering (The Length-Ratio Formula)**

* Koehn, P., et al. (2018). *Findings of the WMT 2018 Shared Task on Parallel Corpus Filtering*. Proceedings of the Third Conference on Machine Translation. *(Cite this to defend your automated mathematical filtering. It establishes that rejecting sentence pairs with extreme length imbalances is an industry-standard baseline for cleaning parallel corpora).*

**LLM Hallucinations (Handling the Junk Columns)**

* Ji, Z., et al. (2023). *Survey of Hallucination in Natural Language Generation*. ACM Computing Surveys, 55(12), 1-38. *(Use this to academically explain why Gemini probabilistically invented extra JSON keys like `nibun_sentence` or outputted long explanations, necessitating your sweeper script).*

**Human Evaluation Sampling (The 100-Word Review)**

* Barrault, L., et al. (2020). *Findings of the 2020 Conference on Machine Translation (WMT20)*. Proceedings of the Fifth Conference on Machine Translation. *(Cite this to support your decision to extract a randomized 100-sentence sample. While automated checks are necessary, this paper reinforces that manual human evaluation of a statistical sample remains the gold standard for establishing ground-truth dataset quality).*

# BorneoSpeak: Offline Neural Machine Translation for the Iban Language

## Project Overview

BorneoSpeak is a mobile-centric digital preservation architecture designed to translate the indigenous Iban language into English without requiring cloud connectivity. By combining zero-latency edge computing with highly optimized Natural Language Processing (NLP), the system provides a multimedia-rich, typography-first interface for users in remote regions. The current architecture successfully implements a static relational database fallback, setting the baseline for the integration of a custom-trained Neural Machine Translation (NMT) engine.

## Phase 1: Data Engineering & Knowledge Distillation

Traditional NLP pipelines fail on underrepresented dialects due to extreme data scarcity. This project overcomes that barrier by bridging foundational secondary data with generative AI to engineer a novel primary dataset.

* **Secondary Data Foundation (Digitization):** Parsed and cleaned legacy OCR scans of Anthony Richards' *An Iban-English Dictionary* (1981), transforming a static, historical secondary source into a master relational database.
* **Derived Primary Data (Synthetic Generation):** Utilized the Gemini 3.1-flash-lite LLM in tightly constrained batch intervals to engineer a novel, synthetically derived primary dataset. This yielded over 6,600 grammatically sound Iban-to-English parallel sentences explicitly anchored to the secondary dictionary parameters.
* **Automated Quality Assurance:** Deployed an algorithmic sweeper to salvage misplaced JSON keys, strip AI hallucinations, and enforce a mathematical length-ratio heuristic (0.3 to 3.5 word-count ratio) to eliminate explanatory anomalies.
* **Human Verification:** A randomized 100-sentence statistical sample evaluated by a fluent speaker achieved an 87.0% semantic accuracy rate, validating this derived primary corpus as a high-quality foundation for neural network training.

## Phase 2: Edge Computing & Interface Architecture

The mobile front-end is engineered for accessibility and zero-latency performance in disconnected environments.

* **Framework:** Developed using Flutter and Dart to ensure cross-platform compatibility and a modern User Experience (UX).
* **Offline Fallback (V1 Baseline):** Currently operational using an embedded SQLite database, proving the edge-computing concept via immediate, offline root-word lookups and local dictionary saves.

## Phase 3: Neural Machine Translation (Pending Integration)

The finalized synthetic dataset will be used to fine-tune a lightweight MarianMT sequence-to-sequence model. This NMT engine will be compiled directly into the application payload to handle unconstrained sentence structures natively on the device, mathematically smoothing the 13% variance observed during the data engineering phase.
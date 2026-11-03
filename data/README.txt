BorneoSpeak: Methodology & Architecture

Project Overview

BorneoSpeak is a mobile-centric digital preservation architecture designed to translate the indigenous Iban language into English without requiring cloud connectivity. By combining zero-latency edge computing with highly optimized Natural Language Processing (NLP), the system provides a multimedia-rich, typography-first interface for users in remote regions. The current architecture successfully implements a static relational database fallback, setting the baseline for the integration of a custom-trained Neural Machine Translation (NMT) engine.

Phase 1: Data Engineering & Knowledge Distillation

Traditional NLP pipelines fail on underrepresented dialects due to extreme data scarcity. This project overcomes that barrier by bridging foundational secondary data with generative AI to engineer a novel primary dataset.

1. Unstructured Data Ingestion and LLM-Assisted Extraction

The pipeline begins with the digitization of Anthony Richards' An Iban-English Dictionary (1981). Because the source material was an OCR-scanned text document containing heavy noise, inconsistent formatting, and artifacts, traditional regex-based scraping was insufficient. A Generative AI extraction script was engineered to read the unstructured text in strategic chunks and map the linguistic data into structured key-value pairs (source_word, english_translation). To navigate API token windows and rate limits, the text was algorithmically sliced and processed in distinct batches, yielding the initial raw CSV datasets.

2. Dataset Sanitization and Master Dictionary Compilation

The raw LLM outputs required programmatic cleaning to become production-ready. A sanitization script was deployed to detect and strip mojibake (encoding errors resulting from the OCR process), remove duplicate entries, and standardize text formatting. Fragmented dataset files were merged into a single, pristine source of truth: cleaned_iban_dataset_MASTER.csv. This file serves as the definitive vocabulary baseline for both the active lookup engine and the machine learning model.

3. Synthetic Parallel Corpus Generation (Knowledge Distillation)

Because Neural Machine Translation (NMT) models require sentence-level context rather than isolated vocabulary, the pipeline utilizes a technique called Knowledge Distillation. A Python automation script feeds batches of the sanitized master dictionary into a cloud-based LLM (Gemini 3.1-flash-lite). The script uses strict prompt engineering to anchor the LLM to the verified dictionary definitions, forcing it to generate grammatically accurate, real-world Iban-to-English sentence pairs while severely reducing hallucination. To ensure the output could be fed directly into a neural network training loop, the prompt enforces strict JSON array formatting, yielding the exact training data required to fine-tune the MarianMT architecture.

4. Automated Quality Assurance & Human Verification

An algorithmic sweeper was deployed to salvage misplaced JSON keys, strip AI hallucinations, and enforce a mathematical length-ratio heuristic (0.3 to 3.5 word-count ratio) to eliminate explanatory anomalies. Finally, a randomized 100-sentence statistical sample evaluated by a fluent speaker achieved an 87.0% semantic accuracy rate, mathematically validating this derived primary corpus as a high-quality foundation for training.

Phase 2: Edge Computing & Interface Architecture

The mobile front-end is engineered for accessibility and zero-latency performance in disconnected environments.

Framework: Developed using Flutter and Dart to ensure cross-platform compatibility and a modern User Experience (UX).

Offline Fallback (V1 Baseline): The master CSV dictionary is compiled into a lightweight, relational SQLite database (.db). This allows the mobile application to query translations locally using SQL indexing, achieving zero-latency lookups entirely offline without relying on cloud infrastructure.

Phase 3: Neural Machine Translation (Pending Integration)

The finalized synthetic dataset will be used to fine-tune a lightweight MarianMT sequence-to-sequence model. This NMT engine will be compiled directly into the application payload (via TFLite/ONNX) to handle unconstrained sentence structures natively on the device, mathematically smoothing the 13% variance observed during the data engineering phase.

Academic References & Literature Review Mapping

The Source Material (The Dictionary)

Richards, A. (1981). An Iban-English Dictionary. Oxford University Press.

The Methodology (Knowledge Distillation & Synthetic Data)

Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. arXiv preprint arXiv:1503.02531.

Wang, Y., et al. (2021). Towards Zero-Label Language Learning. arXiv preprint arXiv:2109.09193.

Google. (2023). Gemini: A Family of Highly Capable Multimodal Models. Google DeepMind.

The Architecture (MarianMT & Edge AI)

Junczys-Dowmunt, M., et al. (2018). Marian: Fast Neural Machine Translation in C++. Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics.

Lin, J., et al. (2020). MCUNet: Tiny Deep Learning on IoT Devices. Advances in Neural Information Processing Systems.

Quality Assurance (Heuristics, Hallucinations, and Sampling)

Koehn, P., et al. (2018). Findings of the WMT 2018 Shared Task on Parallel Corpus Filtering. Proceedings of the Third Conference on Machine Translation.

Ji, Z., et al. (2023). Survey of Hallucination in Natural Language Generation. ACM Computing Surveys, 55(12), 1-38.

Barrault, L., et al. (2020). Findings of the 2020 Conference on Machine Translation (WMT20). Proceedings of the Fifth Conference on Machine Translation.

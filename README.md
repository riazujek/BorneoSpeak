🗣️ BorneoSpeak

Offline Neural Machine Translation for the Iban Language

A mobile-centric digital preservation architecture designed to translate the indigenous Iban language into English without requiring cloud connectivity.

📖 Project Overview

Standard NLP pipelines fail on underrepresented dialects due to extreme data scarcity. BorneoSpeak tackles the digital marginalization of low-resource Pan-Bornean languages by explicitly addressing data scarcity and agglutinative morphological complexity.

By combining zero-latency edge computing with highly optimized Natural Language Processing (NLP), the system provides a multimedia-rich, typography-first interface for users in remote regions without cell service or Wi-Fi.

🏗️ Architecture & Methodology

🟢 Phase 1: Data Engineering & Knowledge Distillation (Completed)

Bridging foundational secondary data with generative AI to engineer a novel, machine-readable primary dataset.

📚 Secondary Data Foundation: Parsed and cleaned legacy OCR scans of Anthony Richards' An Iban-English Dictionary (1981) into a master relational database.

🧠 Derived Primary Data: Utilized the Gemini 3.1-flash-lite LLM via zero-shot Knowledge Distillation. Generated over 6,600 grammatically sound parallel sentences.

🧹 Automated Quality Assurance: Built algorithmic sweeps (clean_and_verify.py) to drop AI hallucinations and enforce a mathematical length-ratio heuristic (0.3 - 3.5x).

🎯 Human Validation: A randomized 100-sentence sample evaluated by a fluent speaker achieved an 87.0% semantic accuracy rate.

Note: Root words from this sample were isolated from the training corpus to prevent data leakage and serve as a zero-shot Gold Test Set.

🟡 Phase 2: Neural Machine Translation (In Progress)

Fine-tuning a sequence-to-sequence model optimized for the morphological rules of the Iban language.

Base Model: Utilizing Helsinki-NLP/opus-mt-ms-en (MarianMT) to leverage linguistic proximity (Malay to English) via transfer learning.

Agglutinative Processing: Employs a Byte-Pair Encoding (BPE) tokenizer to mathematically separate complex Iban affixes (e.g., di-, me-).

A/B Testing: Evaluates NMT performance (BLEU-4) by comparing a model trained on the raw synthetic dataset vs. the heuristically filtered dataset.

🔵 Phase 3: Edge Computing & Interface (Operational Baseline)

The mobile front-end engineered for zero-latency performance in disconnected environments.

Framework: Flutter/Dart for cross-platform compatibility and a modern typography-first UX.

V1 Offline Fallback: Currently operational using an embedded SQLite database, proving the edge-computing concept via immediate, offline static root-word lookups.

V2 Edge AI Deployment (Upcoming): The trained NMT model will be quantized into a lightweight TFLite/ONNX format and embedded directly into the application payload.

📂 Repository Structure

BorneoSpeak/
│
├── data/                  # Datasets & Corpora
│   ├── raw/               # Anthony Richards OCR scans
│   └── final/             # synthetic_parallel_corpus_CLEANED.csv
│
├── scripts/               # Python Pipeline
│   ├── generate_nmt.py    # LLM Knowledge Distillation
│   ├── clean_verify.py    # Automated QA and regex filtering
│   └── train_nmt.py       # HuggingFace MarianMT fine-tuning
│
└── lib/                   # Flutter Application (Front-end)
    ├── main.dart          # Entry point
    └── database/          # SQLite routing and dictionary saving


🚀 Getting Started (V1 Baseline)

To test the mobile application shell and the SQLite offline baseline:

Ensure your mobile emulator is running or a physical device is connected.

Clone the repository and navigate to the root directory.

Install Flutter dependencies:

flutter pub get


Compile and launch the application:

flutter run


(💡 Pro Tip for Panel Evaluation: To test the zero-latency edge architecture, toggle your device into Airplane Mode while running the application).

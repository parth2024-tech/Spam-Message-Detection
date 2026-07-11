# Spam Message Detection

An end-to-end machine learning project to identify whether SMS/text messages are spam or ham (legitimate). The pipeline utilizes text normalization, dual word- and character-level TF-IDF vectorization, a Linear Support Vector Classifier (LinearSVC), sigmoid-based probability calibration, and an optimized decision threshold. 

This repository provides training workflows, evaluation diagnostics, robust testing against text obfuscation, and a command-line interface (CLI) for real-time predictions.

---

## Repository Structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI workflow
├── artifacts/
│   └── README.md              # Documentation for generated model artifacts
├── data/
│   └── raw/
│       └── README.md          # Guide for raw dataset placement
├── tests/
│   └── test_smoke.py          # Pytest smoke tests for the inference path
├── LICENSE                    # MIT License
├── README.md                  # Project documentation
├── predict.py                 # CLI inference script
├── requirements.txt           # Project dependencies
└── sms-spam-detection.ipynb   # Main Jupyter notebook for training and evaluation
```

---

## Installation & Setup

1. **Clone the Repository** (or navigate to the project directory).
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Dataset Setup**:
   This project uses the **SMS Spam Collection** dataset. Place the raw CSV dataset file in the `data/raw/` directory.
   - Expected path: `data/raw/SPAM text message 20170820 - Data.csv`
   - Alternatively, the dataset will be discovered automatically if run in a Kaggle environment using standard Kaggle input directories.

---

## How to Run

### 1. Train and Evaluate the Model
Open and execute the Jupyter notebook:
```text
sms-spam-detection.ipynb
```
The notebook executes the following pipeline:
- **Data Preprocessing & Health Checks**: Drops duplicates, missing entries, and normalizes raw text (e.g., standardizing URLs, email addresses, phone numbers, and monetary figures).
- **Stratified Train/Test Split**: Ensures equal class representation across splits.
- **Feature Extraction**: Generates both word-level and character-level TF-IDF features.
- **Model Training**: Standardizes features and trains a `LinearSVC` classifier.
- **Probability Calibration**: Calibrates decision scores using Platt scaling (sigmoid calibration) so predictions represent true probabilities.
- **Decision Threshold Optimization**: Selects the optimal threshold on cross-validation folds to balance precision and recall.
- **Robustness Checks**: Evaluates how well the model resists simple spelling obfuscations.
- **Artifact Export**: Saves the trained models, feature transformers, metadata, and evaluation results to the `artifacts/` folder.

### 2. Predict Using the CLI
Once the notebook has run and the artifacts are generated in the `artifacts/` folder, you can run inference using `predict.py`.

Predict a single message:
```bash
python predict.py "Hey, are we still meeting for dinner tonight?"
```

Predict a potential spam message:
```bash
python predict.py --text "Congratulations! You won a $1,000 gift card. Claim here."
```

Output predictions in JSON format (ideal for programmatic parsing):
```bash
python predict.py --text "Check this out: free rewards!" --json
```

### 3. Run Automated Tests
A smoke test is provided to verify the end-to-end inference path by training a fast, dummy model and testing CLI responses:
```bash
pytest -q
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

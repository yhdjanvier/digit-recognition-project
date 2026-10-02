# Handwritten Digit Recognition System Using CNN and Streamlit

**Module:** ITLPA701 Python and Fundamentals of AI - **Assessment:** CAT2 Practical 2026
**AI approach:** Deep Learning (Convolutional Neural Network) - **Interface:** Streamlit web app

> **About numbers in this README.** No accuracy / precision / recall / F1 value is written here by hand.
> All metrics are produced by `python train.py` and saved in `results/` (`metrics.json`,
> `classification_report.txt`, `model_comparison.md`). Sections 15-16 therefore say
> **"Generated after running the training pipeline."** Paste your own numbers there after you run it.

---

## 1. Project overview
A user uploads an image of one handwritten digit; a CNN trained on the scikit-learn Digits dataset predicts the digit (0-9), shows the confidence and the probability of every class, and warns when it is unsure.
Assessment workflow demonstrated: **Apply Data Preprocessing -> Apply Deep Learning Algorithms -> Apply Model Evaluation Techniques -> Deployment/Demonstration -> Responsible Use.**

## 2. Problem statement
Handwritten numbers on forms, receipts, answer sheets and records are slow and error-prone to read manually. The system reads a handwritten digit automatically and supports (not replaces) a human reader.

## 3. Objectives
1. Acquire and prepare a dataset of handwritten digit images.
2. Preprocess images into the format a CNN needs.
3. Design and train a CNN that classifies digits 0-9.
4. Evaluate with accuracy, precision, recall, F1-score and a confusion matrix.
5. Improve the model based on the evaluation (evaluate -> adjust -> re-evaluate).
6. Provide a Streamlit application.
7. Discuss limitations, bias, privacy and responsible use.

## 4. Why a CNN?
The input is an image. Convolutional layers share small filters across the image and learn local visual patterns (strokes, curves, loops) regardless of exact position, with far fewer parameters than a fully connected network on raw pixels. RNNs target sequences and FNNs ignore spatial structure, so CNN is the appropriate choice from the permitted options (CNN / RNN / FNN).

## 5. Dataset (data acquisition)
`sklearn.datasets.load_digits()` - bundled with scikit-learn, no manual download, no internet needed after installation.

## 6. Dataset characteristics
Computed by the code (`results/dataset_statistics.json`, printed by `train.py`, shown in the notebook): number of samples, classes, image shape, pixel range, class distribution. Known properties: 1,797 images, 10 classes (digits 0-9), 8x8 grayscale, pixel values 0-16, 64 features per image. Sample images and class distribution are saved as `results/dataset_samples.png` and `results/class_distribution.png`.

## 7. System architecture
```
Uploaded PNG/JPG
   -> grayscale, polarity detection, background removal, crop, stroke normalisation, resize 8x8, centre   (src/preprocessing.py)
   -> CNN: Conv2D -> MaxPool -> Conv2D -> Flatten -> Dense -> Dropout -> Softmax(10)                      (src/model.py)
   -> probabilities for digits 0-9 -> predicted digit + confidence + low-confidence warning               (predict.py)
   -> displayed in Streamlit                                                                               (app.py)
```

## 8. Data preprocessing (Apply Data Preprocessing)
1. Load the dataset; extract `images` (1797, 8, 8) and `labels`.
2. Inspect shape, class distribution and pixel range (programmatically).
3. **Stratified split first** (seed 42): 20 % test; of the rest 20 % validation; remainder training. Done by index so the sets are provably disjoint (tested).
4. **Normalise** pixels with the fixed constant 16 (0-16 -> 0-1). A constant, not data statistics, so nothing leaks from test data.
5. **Reshape** to `(N, 8, 8, 1)` (channel axis for Conv2D).
6. **Labels** remain integers 0-9 and the loss is `sparse_categorical_crossentropy` - mathematically the same as one-hot labels with categorical cross-entropy, but simpler.
7. **Augmentation (training split only):** random +-1 pixel shifts, used by the adjusted model.

**Training/inference consistency:** uploaded images are converted to the same representation as the training data: bright ink = high value on a dark background, longest side of the digit about 8 px, values 0-1 (see section 17).

## 9. Feature engineering
* **Input features:** the raw 8x8 pixels are the initial representation (no hand-crafted features - they would conflict with the CNN idea).
* **Input preparation:** normalisation to 0-1, channel axis, optional shift augmentation.
* **Model-level feature engineering:** Conv layer 1 (32 filters) learns edge/stroke detectors; Conv layer 2 (64 filters) combines them into curves and loops; the Dense layer combines these into class evidence. The notebook visualises the learned feature maps (section 8b).

## 10. CNN architecture
| Layer | Output | Purpose |
|---|---|---|
| Input | 8x8x1 | normalised digit |
| Conv2D 32, 3x3, same, ReLU | 8x8x32 | edges / strokes |
| MaxPooling2D 2x2 | 4x4x32 | keep strongest responses, reduce size |
| Conv2D 64, 3x3, same, ReLU | 4x4x64 | curves / loops |
| Flatten | 1024 | feature vector |
| Dense 64, ReLU | 64 | combine features |
| Dropout (0.30 in the adjusted model) | 64 | reduce overfitting |
| Dense 10, Softmax | 10 | probability per digit |

## 11. Parameter explanation
| Parameter | Value | Reason |
|---|---|---|
| Filters 32 -> 64, kernel 3x3 | | small kernels suit 8x8 images; more filters in the deeper layer for more complex shapes |
| Padding `same` | | stops the already tiny image from shrinking at each convolution |
| One pooling layer | | 8x8 is too small for repeated pooling |
| Dense 64 | | enough capacity, keeps the network small |
| Dropout 0.30 (adjusted) | | regularisation against overfitting |
| Optimizer Adam, lr 0.001 | | adaptive, robust default |
| Loss | sparse categorical cross-entropy | integer labels, 10 classes |
| Batch size 32 | | small dataset - many updates per epoch |
| Epochs | 10 (baseline) / up to 60 with EarlyStopping (adjusted) | see section 16 |
| EarlyStopping (val_loss, patience 8, restore best weights) | | stops when validation loss stops improving |
| ReduceLROnPlateau (x0.5, patience 3) | | smaller steps when progress stalls |
| ModelCheckpoint (best val_loss) | | keeps the best weights on disk (`models/checkpoints/`, git-ignored) |
| Validation | 16 % of all data | monitoring, early stopping, model selection |
| Test | 20 % of all data | final unbiased evaluation only |

## 12. Training process
`python train.py`: loads data -> splits -> trains the **baseline** -> evaluates and diagnoses -> trains the **adjusted** model -> selects the model by **validation** accuracy (ties: lower validation loss) -> evaluates both on the test set (reporting only) -> saves model, configuration, history, plots, metrics.

## 13. Evaluation methodology (Apply Model Evaluation Techniques)
Accuracy, precision, recall, F1 (macro, weighted and per class), confusion matrix, classification report, training/validation accuracy and loss curves. All computed with scikit-learn on the held-out test set.

## 14. Streamlit application
Header and description, upload area (`st.file_uploader`, PNG/JPG/JPEG), preview, **Predict Digit** button, predicted digit, confidence, probability bar chart for 0-9, the 8x8 image the CNN actually sees, low-confidence warning (threshold adjustable), model information and evaluation metrics, responsible-use notice. A "dataset sample" demo mode guarantees a working demonstration even without your own handwriting. The model is loaded once (cached); the app never retrains.

## 15. Actual evaluation results
**Generated after running the training pipeline.** See `results/metrics.json`, `results/classification_report.txt`, `results/confusion_matrix.png`, `results/training_accuracy.png`, `results/training_loss.png`. The interpretation (best/worst class, most-confused digit pairs) is printed by `train.py` / `evaluate.py` from the real numbers.

| Metric (test set) | Value |
|---|---|
| Accuracy | *paste from results/metrics.json* |
| Precision (macro) | *paste* |
| Recall (macro) | *paste* |
| F1-score (macro) | *paste* |

## 16. Evaluation-based adjustment
| | Baseline (initial configuration) | Adjusted (final configuration) |
|---|---|---|
| Epochs | 10 | up to 60, EarlyStopping patience 8 |
| Dropout | 0.0 | 0.30 |
| Augmentation | none | +-1 px shifts (training only) |
| Learning rate | fixed 0.001 | 0.001 with ReduceLROnPlateau |

*Process:* the baseline is trained and its learning curves are diagnosed automatically (`src/training.py::diagnose_history`): train-validation gap, validation loss rising after its minimum (overfitting) and best epoch at the end of training (under-training). The findings are saved in `results/model_comparison.json` / `.md`.
*Reason for the adjustment:* a 10-epoch, un-regularised network on 1,149 training images can be both under-trained (stops early) and prone to overfitting (no dropout); shift augmentation also prepares the model for uploaded images that are never perfectly centred.
*Result:* **generated after running the training pipeline** - `results/model_comparison.md` lists baseline vs adjusted accuracy, precision, recall and F1 on validation and test data. If the adjustment does not improve a metric, the table shows that honestly and the validation rule may keep the baseline; do not edit the numbers.

## 17. Real-world image preprocessing and its limitation
`preprocess_uploaded_image()`: (1) EXIF rotation, transparency -> white, RGB -> grayscale; (2) **polarity detection** from the image border median (dark-on-light vs light-on-dark - images are *not* blindly inverted; the user can override in the sidebar); (3) background subtraction and contrast stretch; (4) noise suppression; (5) crop to the digit; (6) stroke-width normalisation (thin pen lines would disappear when shrunk); (7) scale the longest side to 8 px **keeping the aspect ratio**; (8) place on an 8x8 canvas and centre by centre of mass; (9) scale to 0-1.
*Limitation:* the training data are tiny 8x8 images from a limited group of writers; phone photos have shadows, texture and different stroke styles, so accuracy on uploads is expected to be lower than on the test set. Blank or low-contrast images are rejected with a clear message.

## 18. Responsible AI
**Limitations/risks:** other handwriting styles, unclear/blurred/low-contrast images, unusual digit shapes, wrong preprocessing, over-trusting confidence. **Mitigations:** image instructions, confidence threshold with warning and second-best digit, 8x8 preview, human verification for important decisions, transparent statement that predictions can be wrong, no storage of uploaded images (processed in memory).

## 19. Installation (Windows / PyCharm)
Use **Python 3.10-3.12** (3.11 recommended).
```
cd digit-recognition-project
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
In PyCharm: *Settings -> Project -> Python Interpreter -> Add -> Existing -> `.venv\Scripts\python.exe`*. CPU is enough; no GPU/CUDA needed.

## 20. How to train / evaluate / run
```
python train.py          # trains, adjusts, selects, saves model + results (about 1-3 minutes on CPU)
python evaluate.py       # re-evaluates the saved model on the same test split
streamlit run app.py     # web app at http://localhost:8501
python predict.py my_digit.png      # command-line prediction
pytest -v                # automated tests
```
Notebook: `jupyter notebook notebooks/digit_recognition.ipynb` (or open it in PyCharm), then *Restart & Run All*.

## 21. Project structure
```
app.py  train.py  evaluate.py  predict.py  config.py  requirements.txt  README.md  .gitignore  pytest.ini
models/        digit_cnn_model.keras (created by train.py), model_config.json, baseline_cnn_model.keras
src/           data_loader.py preprocessing.py model.py training.py evaluation.py utils.py
results/       metrics.json classification_report.txt confusion_matrix.png training_accuracy.png training_loss.png
               model_comparison.json/.md dataset_statistics.json dataset_samples.png class_distribution.png ...
notebooks/     digit_recognition.ipynb
tests/         test_project.py
screenshots/   (put your application screenshots here for the submission)
.streamlit/    config.toml
```

## 22. Reproducibility
Fixed seed 42 (Python, NumPy, Keras), deterministic stratified split, pinned-range `requirements.txt`, saved model (`.keras`), saved configuration (`models/model_config.json`: hyper-parameters, class labels, input shape, normalisation, split sizes, library versions), saved results and complete source code. Re-running `python train.py` on another machine gives very similar (not necessarily bit-identical, because of different TensorFlow versions/CPUs) results.

## 23. Saved-model strategy and GitHub
The model is tiny (about 1 MB), so **commit `models/digit_cnn_model.keras`, `models/model_config.json` and `results/` to GitHub**. Streamlit Community Cloud only sees what is in the repository, so the app can load the model without retraining. `.gitignore` excludes `.venv`, caches, IDE files and `models/checkpoints/`.
```
git init
git add .
git commit -m "Handwritten digit recognition - CNN + Streamlit"
git branch -M main
git remote add origin https://github.com/<your-username>/digit-recognition-project.git
git push -u origin main
```
(Run `python train.py` **before** `git add .` so the model file exists.)

## 24. Deployment on Streamlit Community Cloud
1. Push the repository to GitHub (public, or private with Streamlit access granted).
2. Go to https://share.streamlit.io and sign in with GitHub -> **Create app**.
3. Repository: `<your-username>/digit-recognition-project`, branch `main`, main file path `app.py`.
4. *Advanced settings* -> choose **Python 3.11** (TensorFlow does not support every newest Python version).
5. **Deploy.** The first start installs TensorFlow, which takes a few minutes.

## 25. Troubleshooting
| Problem | Fix |
|---|---|
| "Trained model not found" | run `python train.py` first, commit `models/digit_cnn_model.keras` |
| `ModuleNotFoundError: config` / `src` | run commands from the project root; in PyCharm mark the project folder as *Sources Root* |
| TensorFlow will not install | check `python --version` (use 3.10-3.12, 64-bit), upgrade pip |
| Model loads but predictions look wrong for your photo | check the "what the CNN sees" image; set the digit colour manually in the sidebar; use one dark digit on plain light paper filling the picture |
| Streamlit Cloud build fails | choose Python 3.11 and make sure `requirements.txt` is at the repository root |
| Results differ slightly from a classmate | different TensorFlow/CPU; compare against your own `results/` files |

## 26. CAT2 compliance matrix (30 marks)
| Assessment requirement (marks) | Where implemented | Evidence |
|---|---|---|
| Environment: functionalities specified (1.5) | section 19, libraries and their roles | `requirements.txt`, README sections 4, 19 |
| Environment configured (1.5) | venv + pinned ranges, CPU only | `requirements.txt`, `pytest.ini`, `config.py` |
| Data acquired from the given source (3) | `load_digits()` | `src/data_loader.py`, `results/dataset_statistics.json`, `dataset_samples.png` |
| Data pre-processed (3) | split, normalise, reshape, labels, augmentation, upload pipeline | `src/preprocessing.py`, README section 8, `tests/test_project.py` |
| Data features engineered (3) | pixels -> normalised, channel axis, shift augmentation, polarity/crop/stroke/centre for uploads | `src/preprocessing.py`, README sections 8-9 |
| Model features engineered (3) | learned conv feature maps (edges -> curves) | `src/model.py`, notebook section 8b, README section 9 |
| AI approach selected and justified (1.5) | CNN justified for image data | README section 4 |
| Parameters selected and explained (1.5) | table of every parameter with reason | README section 11, `config.py`, `models/model_config.json` |
| Application implemented and tested (6) | CNN, training pipeline, inference, Streamlit, automated tests | `train.py`, `predict.py`, `app.py`, `tests/test_project.py`, notebook |
| Evaluation metrics selected (1) | accuracy, precision, recall, F1, confusion matrix, curves | `src/evaluation.py`, README section 13 |
| Evaluation implemented and interpreted (2) | scikit-learn metrics, generated interpretation | `evaluate.py`, `results/*`, `interpret_metrics()` |
| Parameters adjusted from evaluation (1) | baseline -> diagnosis -> adjusted -> re-evaluation | `src/training.py`, `results/model_comparison.md`, README section 16 |
| Model/configuration saved, reproducible (1) | `.keras` model, config JSON, seed, requirements | `models/`, `config.py`, README section 22 |
| Deployed / demonstrated via web app (1) | Streamlit app | `app.py`, README section 24 |
| Responsible use: limitation + mitigation | confidence warning, human verification, preview | `app.py` (Responsible use tab), README section 18 |

## 27. Document mode (all text: letters, numbers, punctuation) - extension
*Not part of the trained CNN.* A separate tab uses the **pretrained OCR model EasyOCR** to read whole pages (photo, scan or PDF) and show the result as typed, editable text with per-line confidence and a `.txt` download.
**Install:** `pip install -r requirements-ocr.txt` (about 2-3 GB, includes PyTorch; the first run downloads the model weights).
**Test data with ground truth:** `python make_ocr_samples.py` creates 6 printed-style pages (clean and degraded) in `tests/ocr_samples/`, each with a `.txt` containing the exact text. For handwriting, write a few lines on paper, photograph them, and save the photo plus a typed `.txt` of what you wrote under the same name in that folder.
**Evaluate:** `python evaluate_ocr.py` -> Character Error Rate (CER) and Word Error Rate (WER) per page in `results/ocr_evaluation.md` (0 = perfect). These are generated results; none are written here by hand.
**Public datasets for a more serious benchmark:** IAM Handwriting Database (handwritten English lines; free for academic use after registration), FUNSD (199 scanned forms), SROIE (1,000 scanned receipts), EMNIST (single characters).
**Deployment note:** EasyOCR + PyTorch is heavy. Streamlit Community Cloud may run out of memory with it, so keep `requirements.txt` as is (the digit app deploys fine) and use Document mode locally, or on a host with more RAM. If the packages are missing, the tab shows an install message instead of crashing.

## 28. Suggested demonstration order
1. Show dataset samples and statistics -> 2. explain preprocessing and feature engineering -> 3. show architecture (`model.summary()`) -> 4. show training curves -> 5. show metrics, report, confusion matrix, baseline-vs-adjusted table -> 6. Streamlit: upload a digit, show prediction, confidence and probabilities -> 7. explain one limitation and its mitigation.

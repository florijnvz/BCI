# BCI Project

## Project Overview
EEG-based brain-computer interface classification pipeline using the P300 paradigm. Raw EEG data is preprocessed into epochs, features are extracted per subject/session, and a classifier is trained and evaluated across subjects. Subjects 1, 7, 11, 18, 27, 29 are excluded due to data quality issues.

## Data Setup
1. Download the dataset from [zenodo.org/records/3266930](https://zenodo.org/records/3266930)
2. Unzip subjects `subject_02` through `subject_43` (excluding `subject_27`) into `data/raw/`:
```
data/raw/
├── subject_02/
├── subject_03/
...
└── subject_43/
```

## Environment Setup
```bash
pip install -r requirements.txt
```

## Run Order
```
1. python 00_batch_preprocessing.py       # preprocesses raw EEG → data/preprocessed/
2. python 00_batch_feature_extraction.py  # extracts features → data/preprocessed/
3. jupyter notebook 03_classification.ipynb  # runs classification & analysis
```
`01_preprocessing.ipynb` and `02_feature_extraction.ipynb` are single-subject development notebooks; use the batch scripts for the full pipeline.

## Subject Exclusions
The following subjects are skipped automatically by the batch scripts — no manual action needed:

| Subject | Reason |
|---------|--------|
| 1 | Excluded |
| 7 | Excluded |
| 11 | Excluded |
| 18 | Excluded |
| 27 | Not in dataset |
| 29 | Excluded |

## Project Structure
```
BCI_project/
├── 00_batch_preprocessing.py       # batch preprocessing script
├── 00_batch_feature_extraction.py  # batch feature extraction script
├── 01_preprocessing.ipynb          # single-subject preprocessing notebook
├── 02_feature_extraction.ipynb     # single-subject feature extraction notebook
├── 03_classification.ipynb         # classification & statistical analysis
├── requirements.txt
├── README.md
└── data/
    ├── raw/                        # downloaded dataset (not tracked)
    └── preprocessed/
        ├── preprocessing_log.txt
        ├── feature_extraction_log.txt
        ├── subject_XX_session_YY_epo.fif
        └── subject_XX_session_YY_features.npz
```

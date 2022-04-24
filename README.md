CS598 Projects: Deep Learning for Healthcare Final Project
==============================

Team: 
- Edgar Pino <edgarsp2@illinois.edu>
- Ivan Zlatev <izlat2@illinois.edu>

Final project code data paper and presentation. 

Project Setup
------------
1. Create conda environment: `make create_environment`.
1. Activate conda env: `source .venv/bin/activate`.
1. Install pip requirements: `make requirements`.


### Model Scripts
1. `cd` into `src/models`
1. Trining: `python ./train_model.py`
1. Validating: `python ./train_model.py`
1. Testing: `python ./train_model.py`. A file inside `src/results/test-output.json` will be generated with predicted and actual queries. 

### Generating SQLite and test data files from sampling
1. Download full `MIMIC-III Clinical Database`: 
    - Demo: https://physionet.org/content/mimiciii-demo/1.4/
    - Full: https://physionet.org/content/mimiciii/1.4/
1. Unzip and place all CSV files inside `src/data/mimic_iii/files`.
1. `cd src/data`
1. Run the `make_dataset.py`: 
    - `python make_dataset.py ./mimic_iii ./mimic_iii_processed/`
1. This should generated the needed SQLite files to run the generated queries from the model. 


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

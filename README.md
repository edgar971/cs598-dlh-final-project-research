CS598 Projects: Deep Learning for Healthcare Final Project
==============================

Team: 
- Edgar Pino <edgarsp2@illinois.edu>
- Ivan Zlatev <izlat2@illinois.edu>

Final project code data paper and presentation. 

### Original work reproduced
Ping Wang, Tian Shi, and Chandan K. Reddy. "Text-to-SQL Generation for Question Answering on Electronic Medical Records." In Proceedings of The Web Conference 2020 (WWW’20), pp. 350-361, 2020.

```
@inproceedings{wang2020text,
  title={Text-to-SQL Generation for Question Answering on Electronic Medical Records},
  author={Wang, Ping and Shi, Tian and Reddy, Chandan K},
  booktitle={Proceedings of The Web Conference 2020},
  pages={350--361},
  year={2020}
}
```
Original work repository: https://github.com/wangpinggl/TREQS

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

## Reproduction Results

<table>
  <thead>
    <tr>
      <th>Questions Dataset</th>
      <th colspan="2">Reproduction Results</th>
      <th colspan="2">Original Work Results</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td></td><td>Acc_LF</td><td>Acc_EX</td><td>Acc_LF</td><td>Acc_EX</td>
    </tr>
    <tr>
      <td>Template</td><td>0.595</td><td>0.756</td><td>0.802</td><td>0.825</td>
    </tr>
    <tr>
      <td>Template + recover</td><td>0.239</td><td>0.394</td><td>0.912</td><td>0.940</td>
    </tr>
    <tr>
      <td>NL</td><td>0.442</td><td>0.700</td><td>0.486</td><td>0.556</td>
    </tr>
    <tr>
      <td>NL + recover</td><td>0.165</td><td>0.352</td><td>0.556</td><td>0.654</td>
    </tr>
  </tbody>
</table>

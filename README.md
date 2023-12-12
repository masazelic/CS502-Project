# CS502-Project

This repository is coding part of the Few-Shot Learning benchmark for the Biomedical datasets, developed by [Brbic Lab](https://brbiclab.epfl.ch/) and extended by Marija Zelic and Elena Mrdja with Relation Network algorithm.  

## Instalation  

For successful running of our benchmark locally, first clone this repository.  

### Conda

Create a conda environment and install it with following command:

```bash
conda env create -f environment.yml 
```

Before each run, activate the environment with:

```bash
conda activate few-shot-benchmark 
```

### Pip

Alternatively, for environments that do not support
conda (e.g. Google Colab), install requirements with:

```bash
python -m pip install -r requirements.txt
```  

## Usage

### Training  

For the hyperparameter tuning of the RelationNet model for the specific problem and dataset, first go to the '''hyperparameter_tuning.py''' and inside '''main()''' function change n_way, n_suport, n_query and dataset name ('swissprot' or 'tabula_muris') and then run following command:  

```bash
python hyperparameter_tuning.py
```

The training process will automatically evaluate at the end. To only evaluate without
running training, use the following:

```bash
python run.py exp.name={exp_name} method=maml dataset=tabula_muris
```

By default, method is set to MAML, and dataset is set to Tabula Muris.
The experiment name must always be specified.  

### Testing

The training process will automatically evaluate at the end. To only evaluate without
running training, use the following:

```bash
python run.py exp.name={exp_name} method=maml dataset=tabula_muris mode=test
```
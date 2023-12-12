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

For the hyperparameter tuning of the RelationNet model for the specific problem and dataset, first go to the `hyperparameter_tuning.py`. Inside `if __name__ == '__main__':` function change parameters n_way, n_suport, n_query and dataset name (`swissprot` or `tabula_muris`) according to your problem and then run following command:  

```bash
python hyperparameter_tuning.py
```  
By default, hyperparameter tuning is set to the Swiss-Prot dataset, 5-way, 5-shot, 15-query problem. It is also important keep line `wandb.log({"loss": avg_loss / float(i + 1)})` in the `meta_template.py`'s `train_loop()` function commented, during this execution.

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

Run `run.py` with the same parameters as the training run, with `mode=test` and it will automatically use the
best checkpoint (as measured by val ACC) from the most recent training run with that combination of
exp.name/method/dataset/model. To choose a run conducted at a different time (i.e. not the latest), pass in the timestamp
in the form `checkpoint.time="'yyyymmdd_hhmmss'"` To choose a model from a specific epoch, use `checkpoint.iter=40`. 
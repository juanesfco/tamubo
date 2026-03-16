# Clone and Environment Setup

```bash
git clone https://github.com/juanesfco/tamubo.git
cd tamubo
conda env create -f envs/exactbo.yml
conda activate tamubo-exactbo
pip install -U pip
pip install -e .
```

## Run Experiment

```bash
python experiments/exactbo/experiment2/run_seed_sweep.py
```

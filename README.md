## Running command

```bash
# lj13 dataset experiment 
uv run python -u -m tr_asbs.experiment.train_asbs experiment=lj13 # with asbs
uv run python -u -m tr_asbs.experiment.train_gamma_asbs experiment=lj13
uv run python -u -m tr_asbs.experiment.train_gamma_asbs_iw experiment=lj13 gamma_schedule=fixed
uv run python -u -m tr_asbs.experiment.train_gamma_asbs_iw experiment=lj13 gamma_schedule=trust_region
```



# gmm varying-dimension experiment
```bash
uv run python -u -m tr_asbs.experiment.train_asbs experiment=gmm_dim
uv run python -u -m tr_asbs.experiment.train_gamma_asbs experiment=gmm_dim
uv run python -u -m tr_asbs.experiment.train_gamma_asbs_iw experiment=gmm_dim gamma_schedule=fixed
uv run python -u -m tr_asbs.experiment.train_gamma_asbs_iw experiment=gmm_dim gamma_schedule=trust_region
Copy code

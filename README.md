# THOC-Pytorch

(Unoffical) Implementation of THOC model : [Lifeng Shen, Zhuocong Li, James T. Kwok:
Timeseries Anomaly Detection using Temporal Hierarchical One-Class Network. NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/97e401a02082021fd24957f852e0e475-Paper.pdf).

If you have noticed errors in implementing, or found better hyperparamters/scores, plz let me know via github issues, pull request, or whatever communication tools you'd prefer.

## Quickstart

```sh
touch secret.py < echo "WANDB_API_KEY={your_wandb_api_key}" # this repo utilizes wandb.
sh script.sh {gpu1} {gpu2} {gpu3} {gpu4}
```

## Wandb Sweep
```sh
wandb sweep hptune/{data_name}.yaml
CUDA_VISIBLE_DEVICES={gpu_id} wandb agent {sweep_id}
```

## Re-implementation so far
THOC uses F1-PA metrics for evaluation: 

|                | Paper | Our Repo |
|----------------|-------|----------|
| NeurIPS-TS-MUL |   -   |    -     |
| SWaT           |   -   |    -     |
| MSL            |   -   |    -     |
| SMAP           |   -   |    -     |


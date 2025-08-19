# RL-SMPC
## Mamba (Conda) env
```
mamba env create -n rlsmpc python=3.10
```
add `alias mam_smpc='cd ~/<your_workspace> && mamba activate rlsmpc && export PYTHONPATH="/home/<your_name>/miniforge3/envs/rlsmpc/lib/python3.10/site-packages:$PYTHONPATH" && export PYTHONPATH=$PYTHONPATH:/home/<your_name>/<your_workspace>/src/f1tenth_gym'` to ~/.bashrc
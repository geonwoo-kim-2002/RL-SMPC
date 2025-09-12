## Start with mamba (conda)
```bash
mamba env create -n rlsmpc python=3.10
echo 'alias mam_rlsmpc='\''cd ~/<your_workspace> && mamba activate rlsmpc && export PYTHONPATH="/home/<your_name>/miniforge3/envs/rlsmpc/lib/python3.10/site-packages:$PYTHONPATH" && export PYTHONPATH=$PYTHONPATH:/home/<your_name>/<your_workspace>/src/f1tenth_gym'\''' >> ~/.bashrc
```
</br>

**Install Dependencies**
```bash
cd src/ &&
git clone https://github.com/geonwoo-kim-2002/f1tenth_gym.git

mam_rlsmpc
pip install -r src/RL-SMPC/requirements.txt
```
</br>

**Install messages**
```bash
cd ~/<your_workspace>/src
git clone https://github.com/geonwoo-kim-2002/pred_msgs.git
git clone https://github.com/geonwoo-kim-2002/RL-SMPC_srv.git
```

## Getting Started
```bash
mam_rlsmpc
colcon build --symlink-install
source install/setup.bash
mkdir models && mkdir videos
ros2 run rl_switching_mpc training
```
</br>

### ❌ Error
```md
--- stderr: f1tenth_gym_ros
/home/a/miniforge3/envs/f1tenth_gym_ros/lib/python3.10/site-packages/setuptools/_distutils/dist.py:289: UserWarning: Unknown distribution option: 'tests_require'
  warnings.warn(msg)
usage: setup.py [global_opts] cmd1 [cmd1_opts] [cmd2 [cmd2_opts] ...]
   or: setup.py --help [cmd1 cmd2 ...]
   or: setup.py --help-commands
   or: setup.py cmd --help

error: option --editable not recognized
---
```
**solution**
```bash
pip install --upgrade "setuptools<66"
```
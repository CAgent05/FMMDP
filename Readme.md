# FMMDP

This is the algorithm code and experimental results for the paper **FMMDP: Failure Monitoring Approach for DNN-based Markov Decision Process**

This work is a further extension of **DRLFailureMonitor**, which leverages the representational capabilities of deep neural networks to expand the original framework, which was solely focused on **Deep Reinforcement Learning Failure Monitoring**, to a broader range of **MDP** tasks.

## Installation
* Bulid from source code
```bash
git clone 
cd FMMDP
conda create -n FMMDP python=3.8
conda activate FMMDP
pip install -r requirements.txt
```

## Steps to run the code

### 1. Training Data collection
For **sensor**-based environment and agent
```python
python DataCollection.py --dataset BipedalWalkerHC --episodes 3000 --nsteps 20
```
* **dataset:** BipedalWaklkerHC, Hopper, InvertedDoublePendulum, Walker2d, Humanoid.
* **episodes:** Number of episodes to collect the data. The default value is 3000.
* **nsteps:** Number of steps in each episode. When nsteps is 20, we also save state-only and state-action-reward time series for comparison with state-action data.

For **vision**-based environment and agent
```python
python DataCollection4CarRacing.py --method FMMDP --episodes 3000 --nsteps 20
```
* **method:** FMMDP, Thirdeye
* **episodes:** Number of episodes to collect the data. The default value is 3000.
* **nsteps:** Number of steps in each episode. When nsteps is 20, we also save state-only and state-action-reward time series for comparison with state-action data.

### 2.Model Training 
```python
python Todynet/src/train.py --dataset BipedalWalkerHCSA --nsteps 20 --epochs 100
```
* **dataset:** use Dataset ( BipedalWaklkerHC, Hopper, InvertedDoublePendulum, Walker2d, Humanoid, CarRacing ) + SAR ( S-State, A-Action, R-Reward ). 
* **epochs:** Number of epochs to train the model. The default value is 100.

### 3. Online Monitor
```python
python TodyNet/OnlineMonitor.py --dataset BipedalWalkerHCSA --nsteps 20
```
* Parameter

### 4. Data Analysis
```python
python result/DataAnalysis.py --dataset BipedalWalkerHC --nsteps 20 --alg TodyNet
```

## Comparison with Thirdeye
### 1. Data Collection
```python
python Thirdeye/DataCollection.py
```
### 2. Online Monitoring
```python
python Thirdeye/thirdeye.py -s test
```

### 3. Result Analysis
```python
python Thirdeye/thirdeye.py -s analyze
```

## TodyNet vs. Alternative MTSC Methods

### MLSTM-FCN
```python
python MLSTM-FCN/train.py --dataset BipedalWalkerHCSA --nsteps 20 --epochs 100
# Sensor-based 
python MLSTM-FCN/OnlineMonitor.py --dataset BipedalWalkerHCSA --nsteps 20
# Vision-based
python MLSTM-FCN/OnlineMonitor4CarRacing.py --dataset BipedalWalkerHCSA --nsteps 20
```

### OS-CNN
```python
python OS-CNN/train.py --dataset BipedalWalkerHCSA --nsteps 20 --epochs 100
# Sensor-based 
python OS-CNN/OnlineMonitor.py --dataset BipedalWalkerHCSA --nsteps 20
# Vision-based
python OS-CNN/OnlineMonitor4CarRacing.py --dataset BipedalWalkerHCSA --nsteps 20
```

### WEASEL
```python
python WEASEL_MUSE/train.py --dataset BipedalWalkerHCSA --nsteps 20
# Sensor-based 
python WEASEL_MUSE/OnlineMonitor.py --dataset BipedalWalkerHCSA --nsteps 20
# Vision-based
python WEASEL_MUSE/OnlineMonitor4CarRacing.py --dataset BipedalWalkerHCSA --nsteps 20
```

**dataset:** use Dataset ( BipedalWaklkerHC, Hopper, InvertedDoublePendulum, Walker2d, Humanoid, CarRacing ) + SA ( S-State, A-Action). 

**epochs:** Number of epochs to train the model. The default value is 100.

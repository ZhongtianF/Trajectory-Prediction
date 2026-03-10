# Human Trajectory Prediction with Scene Context

**T2IA Project — EFREI Paris**

Author: **Zhongtian FAN**

This project studies **pedestrian trajectory prediction** using deep learning models on the ETH/UCY benchmark datasets.

The goal is to predict the **future positions of pedestrians** based on their past observed trajectories and scene context.

---

# Models Implemented

The following models are implemented and compared:

- Constant Velocity baseline
- LSTM Encoder–Decoder
- Scene-aware LSTM (with visual context)
- Conditional GAN for multimodal prediction

The GAN model generates **multiple possible future trajectories** to model the uncertainty of human motion.

---

# Dataset

We use the **ETH / UCY pedestrian trajectory datasets**, which contain real-world pedestrian movements in outdoor scenes.

Scenes used in this project:

- ETH
- HOTEL
- UNIVERSITY
- ZARA01
- ZARA02

Each trajectory sample contains:

Observed trajectory: **8 timesteps**

Future trajectory: **12 timesteps**

Coordinates are represented as **(x, y) positions** of pedestrians.

The trajectory data is stored in **SQLite databases**.

Example dataset table:

```
dataset_T_length_20delta_coordinates
```

Columns:

```
pos_x
pos_y
pos_x_delta
pos_y_delta
ped_id
frame_num
data_id
```

---

# Project Structure

```
Trajectory-Prediction
│
├── src
│   ├── baseline_cv.py
│   ├── config.py
│   ├── dataset.py
│   ├── eval_lstm.py
│   ├── eval_gan.py
│   ├── gan_models.py
│   ├── train_lstm.py
│   ├── train_gan.py
│   ├── visualize.py
│   └── visualize_gan.py
│
├── runs
│   ├── lstm
│   ├── vis
│   └── vis_gan
│
├── requirements.txt
├── README.md
└── report.pdf
```

---

# Installation

Create a Python environment

```
python -m venv .venv
source .venv/bin/activate
```

Install dependencies

```
pip install -r requirements.txt
```

If needed, install manually:

```
pip install numpy pandas matplotlib tqdm pillow opencv-python
pip install torch torchvision
```

---

# Running the Models

### Constant Velocity Baseline

```
python -m src.baseline_cv
```

---

### Train LSTM

```
python -m src.train_lstm --scene eth
```

### Evaluate LSTM

```
python -m src.eval_lstm --scene eth
```

---

### Train GAN

```
python -m src.train_gan --scene eth
```

### Evaluate GAN (Best-of-K)

```
python -m src.eval_gan --scene eth --K 20
```

---

# Visualization

Visualize predicted trajectories:

```
python -m src.visualize
```

Multimodal GAN predictions:

```
python -m src.vis_multimodal --scene eth --index 0 --K 20
```

Overlay trajectories on scene image:

```
python -m src.vis_overlay --scene eth --index 0
```

---

# Evaluation on All Scenes

To evaluate all scenes automatically:

```
python -m src.eval_all_scenes
```

Results are saved to:

```
runs/summary/metrics_val_K20.csv
```

---

# Evaluation Metrics

The project uses standard trajectory prediction metrics.

### ADE (Average Displacement Error)

Average Euclidean distance between predicted trajectory and ground truth trajectory.

### FDE (Final Displacement Error)

Distance between predicted final position and ground truth final position.

For stochastic GAN models:

### Best-of-K ADE / FDE

Minimum error among K generated trajectories.

### Diversity

Average pairwise distance between generated trajectories.

---

# Experimental Results

Overall model performance:

| Model | ADE | FDE |
|------|------|------|
| Constant Velocity | 0.5493 | 1.2102 |
| LSTM | 0.6276 | 1.2192 |
| GAN (single sample) | 0.6576 | 1.2749 |
| **GAN (Best-of-20)** | **0.5422** | **1.0610** |

The results show that deterministic LSTM models do not outperform the simple Constant Velocity baseline. However, the GAN model achieves the best performance when evaluated using **Best-of-K sampling**, demonstrating the advantage of multimodal trajectory prediction.

---

# References

ETH / UCY dataset  
Pellegrini et al.  
**You'll Never Walk Alone: Modeling Social Behavior for Multi-target Tracking**

Introvert paper  
Shafiee et al.  
**Human Trajectory Prediction via Conditional 3D Attention (CVPR 2021)**

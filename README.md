# ATARI Pong DQN

Train a Deep Q-Network (DQN) agent to play Atari Pong, evaluate from a saved checkpoint, and run everything either locally or in Google Colab with the provided notebook.

This repository contains:
- A minimal PyTorch DQN implementation for Pong (Gymnasium ALE).
- Training script (main.py) and evaluation script (eval.py).
- A Colab-ready notebook (ATARI.ipynb) to train or load a checkpoint and play interactively.
- Shared utilities for configuration, checkpoints, preprocessing, and replay buffer.


## Referenced Work (Paper)
This project is based on the seminal DQN work by DeepMind:
- Mnih et al., Playing Atari with Deep Reinforcement Learning, arXiv:1312.5602 (2013) https://arxiv.org/abs/1312.5602
BibTeX (2013 preprint):
```
@article{mnih2013playing,
  title   = {Playing Atari with Deep Reinforcement Learning},
  author  = {Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Graves, Alex and Antonoglou, Ioannis and Wierstra, Daan and Riedmiller, Martin},
  journal = {arXiv preprint arXiv:1312.5602},
  year    = {2013}
}
```


## Project Structure
- main.py — training loop for DQN on ALE/Pong-v5
- eval.py — load a checkpoint and play episodes (renders, optionally saves frames)
- nn.py — DQN network definition
- helper.py — frame preprocessing and 4-frame state stacking
- rpbuf.py — simple replay buffer
- checkpoint.py — save/load checkpoints (policy, target, optimizer, frame index)
- config.py — shared constants (DEVICE, directories, action mapping)
- ATARI.ipynb — Colab notebook to train or evaluate interactively
- models/ — default directory where checkpoints are stored
- observations/ — where raw frames are saved during eval (if using eval.py)


## Requirements
- Python 3.9+ recommended
- See requirements.txt for Python packages
  - gymnasium[atari] >= 0.29.1
  - ale-py and AutoROM (to install Atari ROMs)
  - torch, numpy, opencv-python, pillow, ipywidgets, etc.


## Local Setup (Windows/Mac/Linux)
1) Create and activate a virtual environment
- Windows (PowerShell):
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
- macOS/Linux (bash):
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) Install Atari ROMs (required by ALE)
```bash
python -m AutoROM --accept-license
```
If you’ve already installed the ROMs on your machine before, this step may report that everything is satisfied.

4) Train
```bash
python main.py
```
- Training automatically creates the models/ directory and periodically saves checkpoints like models/dqn_pong_model_10000.pth.
- To resume from the latest checkpoint, set the variable CHECKPOINT_OPTION near the top of main.py to "latest" (default). To start fresh, set it to "none". You can also set a specific path like "models/dqn_pong_model_100000.pth".

5) Evaluate
```bash
python eval.py --checkpoint latest --episodes 3
```
- Options for --checkpoint:
  - latest — loads newest file in models/
  - none — runs a randomly initialized policy
  - A specific path, e.g., models/dqn_pong_model_100000.pth
- During eval, raw observation frames are saved to observations/ as PNGs; you can inspect them later.


## Using Google Colab (Recommended for GPU)
The notebook ATARI.ipynb is designed to be run directly in Colab. It will clone this repo and install dependencies automatically.

Steps:
1) Open the ATARI.ipynb in [Google Colab](https://colab.research.google.com/drive/12PDpDMbz9w77K_MU4feKWB1SZlII8n99?usp=sharing)
2) Run the first cells to clone the repo and install requirements. The notebook attempts to run:
   - `pip install -r requirements.txt`
   - `python -m AutoROM --accept-license`
3) In the notebook’s configuration cell, set:
   - MODE = 'train' to train or resume
   - MODE = 'eval' to load a checkpoint and play episodes
   - CHECKPOINT_OPTION = 'none' | 'latest' | '<path>'
   - EVAL_EPISODES for the number of episodes when in eval mode
4) If you have checkpoints in Google Drive, mount Drive in the notebook and point CHECKPOINT_OPTION to the file path, or copy the file into /content/ATARI/models.
    By default, one model checkpoint is provided in the repository and can be loaded using the 'latest' flag
5) Make sure your Colab runtime has a GPU: Runtime > Change runtime type > Hardware accelerator: GPU.

Saved files in Colab:
- Checkpoints: /content/ATARI/models (the notebook prepends "./ATARI/" to CHECKPOINT_DIR when running in Colab)
- If evaluating with inline rendering, frames render directly in the notebook.


## Configuration and Checkpoints
- Shared constants live in config.py:
  - DEVICE — CUDA if available
  - CHECKPOINT_DIR — default "models"
  - VALID_ACTIONS — [0, 2, 3] for NOOP, UP, DOWN, this is specifically for PONG, other ATARI games use different actions
- Checkpoint behavior is centralized in checkpoint.py with:
  - save_checkpoint(policy, target, optimizer, frame_idx)
  - load_checkpoint(policy, device, option, target_net=..., optimizer=...)
- main.py uses CHECKPOINT_OPTION (string) to decide how to start:
  - "none": start fresh
  - "latest": resume from newest checkpoint in models/
  - path: load that exact file
- eval.py takes --checkpoint and --episodes as CLI arguments.


## Tips and Troubleshooting
- If ALE reports missing ROMs, re-run: `python -m AutoROM --accept-license`.
- If you see import errors for ale_py or gymnasium[atari], update pip packages per requirements.txt.
- Training is slow on CPU; using a GPU is highly recommended.
- Checkpoints can be large; periodically clean models/ if storage is tight.


## Acknowledgments
- This project is inspired by the original DQN implementation by DeepMind and many open-source re-implementations in PyTorch.

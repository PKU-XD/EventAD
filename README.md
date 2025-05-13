# When Every Millisecond Counts: Real-Time Anomaly Detection via the Multimodal Asynchronous Hybrid Network

Anomaly detection is essential for the safety and reliability of autonomous driving systems. Current methods often focus on detection accuracy but neglect response time, which is critical in time-sensitive driving scenarios. In this paper, we introduce real-time anomaly detection for autonomous driving, prioritizing both minimal response time and high accuracy. We propose a novel multimodal asynchronous hybrid network that combines event streams from event cameras with image data from RGB cameras. Our network utilizes the high temporal resolution of event cameras through an asynchronous Graph Neural Network and integrates it with spatial features extracted by a CNN from RGB images. This combination effectively captures both the temporal dynamics and spatial details of the driving environment, enabling swift and precise anomaly detection. Extensive experiments on benchmark datasets show that our approach outperforms existing methods in both accuracy and response time, achieving millisecond-level real-time performance.

![System Overview](assets/framework.jpg)

## Installation Requirements

- Python 3.7+
- PyTorch 1.7+
- PyTorch Geometric
- NumPy
- Matplotlib
- scikit-learn
- tqdm
- h5py


## Installation

First, download the github repository and its dependencies
```bash
WORK_DIR=/path/to/work/directory/
cd $WORK_DIR
git clone git@github.com:PKU-XD/EventAD.git
EVENTAD_DIR=$WORK_DIR/EventAD
cd $EVENTAD_DIR
```

Then start by installing the main libraries. Make sure Anaconda (or better Mamba), PyTorch, and CUDA is installed.
```bash
cd $DAGR_DIR
conda create -y -n EventAD python=3.8
conda activate EventAD
conda install -y setuptools==69.5.1 mkl==2024.0 pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

Then install the pytorch-geometric libraries. This may take a while.
```bash
bash install_env.sh
```

The above bash file will figure out the CUDA and Torch version, and install the appropriate pytorch-geometric packages. Then, download and install additional dependencies locally
```bash
bash download_and_install_dependencies.sh
conda install -y h5py blosc-hdf5-plugin
```

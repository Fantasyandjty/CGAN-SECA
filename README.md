# CGAN-SECA

PyTorch implementation of CGAN-SECA, a time-domain  speech enhancement model integrating a squeeze-and-excitation channel attention mechanism into a conditional generative adversarial network.

The generator uses a symmetric encoder-decoder architecture with skip connections and SECA modules. The discriminator uses SECA, Virtual Batch Normalization (VBN), and Dropout. The model is trained using a least-squares adversarial objective and an L1 waveform reconstruction loss.

## Pre-requisites

The code was developed with the following environment:

- Python 3.10.11
- Windows 11
- PyTorch 2.5.1+cu118
- CUDA 11.8
- NVIDIA GeForce RTX 3080 Ti
- NumPy 2.2.6
- SciPy 1.15.3
- Librosa 0.11.0
- SoundFile 0.14.0
- pystoi 0.4.1

Install the Python dependencies:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```



## Dataset Preparation

Download the [VoiceBank-DEMAND dataset](https://doi.org/10.7488/ds/2117). Resample the waveform files to 16 kHz and organize the clean and noisy speech as follows:

```text
cganseca/
└── voicedemand_16k/
    ├── train/
    │   ├── clean/
    │   └── noisy/
    └── test/
        ├── clean/
        └── noisy/
```

The original VoiceBank-DEMAND directories can be mapped as follows:

- `clean_trainset_28spk_wav` → `voicedemand_16k/train/clean`
- `noisy_trainset_28spk_wav` → `voicedemand_16k/train/noisy`
- `clean_testset_wav` → `voicedemand_16k/test/clean`
- `noisy_testset_wav` → `voicedemand_16k/test/noisy`

Each clean waveform and its noisy counterpart must have the same filename. The scripts load the audio as mono speech at 16 kHz.

Generate the 16,384-sample training segments with a segment shift of 8,192 samples:

```bash
cd cganseca
python data_geneation.py
```

This command generates the cached waveform segments and the training file list:

```text
cache/
scp/train_segan.scp
```

## Training

Start the main CGAN-SECA experiment with the default configuration in `hparams.py`:

```bash
cd cganseca
python train.py --device cuda:0
```

The default main-experiment settings are:

| Setting | Value |
| --- | ---: |
| Sampling rate | 16 kHz |
| Segment length | 16,384 samples |
| Segment shift | 8,192 samples |
| Epochs | 100 |
| Minibatch size | 64 |
| VBN reference batch size | 128 |
| Generator learning rate | 2 × 10⁻⁴ |
| Discriminator learning rate | 2 × 10⁻⁴ |
| Optimizer | RMSprop |
| L1 reconstruction weight | 100 |

Generator checkpoints are saved under:

```text
save/SECA_VBN_DO_improved_s1234/
```



## Main Files

```text
cganseca/
├── data_geneation.py   # dataset segmentation and training list generation
├── dataset.py          # dataset loading and pre-emphasis
├── hparams.py          # main experiment configuration
├── model.py            # CGAN-SECA generator and discriminator
├── losses.py           # adversarial and reconstruction losses
├── train.py            # model training
├── evaluate.py         # speech enhancement and objective evaluation
├── metrics.py          # evaluation metrics
├── complexity.py       # parameter count and inference-time analysis
└── stats_test.py       # statistical significance tests
```


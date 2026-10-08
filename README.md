# Noise Suppression and Velocity Reconstruction for Particle Streak Velocimetry

![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![License](https://img.shields.io/badge/license-MIT-green)

Code, data tables and figures for the paper **"An Integrated Image-Processing and CNN-Based Pipeline for Noise Suppression and Velocity Reconstruction in Particle Streak Velocimetry"** (submitted to *Engineering Research Express*).

**Authors:** Abdul Qadir (Wichita State University, USA), Luqman Razzaq (University of Gujrat, Pakistan), Jabir Ali Siddique (University of Science and Technology, Daejeon, South Korea) and Ramazan Asmatulu (Wichita State University, corresponding author).

---

## What this project does

Particle Streak Velocimetry (PSV) reads fluid velocity from the length of the streaks that tracer particles leave during one exposure. Real PSV frames are messy: static background from the channel walls, camera noise, specular reflections, out-of-focus halos and overlapping streaks all look a bit like streaks. This pipeline cleans a frame and turns what is left into velocity vectors in five stages:

1. **Background subtraction** using the average of 50 flow-off reference frames
2. **Adaptive contrast enhancement and thresholding** at the 85th percentile of the non-zero pixels, so no manual retuning between pulsed and continuous lighting
3. **Contour-based candidate extraction** with bounding boxes kept for later mapping
4. **CNN classification** of each 224 x 224 candidate as *retained* or *replaced*
5. **Reconstruction and velocity extraction**: replaced regions go back to local background, retained streaks give length, orientation and speed

The CNN is deliberately modest in role. Background averaging, thresholding and area filtering do most of the noise suppression, and the CNN is the last discriminative step. The contribution is the integration and experimental validation of the whole workflow, not a new learning algorithm.

![Pipeline](results/paper_figures/Fig01_preprocessing_pipeline.png)

## Headline results from the paper

Experiment: laminar flow in a 60 x 5 x 0.9 mm PDMS microchannel, Re = 2.4, 20 um polystyrene tracers, 20x objective, 45 ms exposure.

| Quantity | Result |
|---|---|
| Test-set accuracy (300 unseen patches) | 99.7 % |
| Precision / recall / F1 (retained = positive) | 1.000 / 0.993 / 0.997 |
| Confusion matrix | 150 of 150 replaced and 149 of 150 retained correct |
| Mean maximum-velocity-normalised absolute error (MVNAE) | 2.64 % |
| RMS error / maximum pointwise error | 3.19 % / 7.28 % |
| Profile positions with MVNAE at or below 5 % | 19 of 20 |
| Processing time (CPU, whole pipeline) | 148 ms per frame |

MVNAE is the absolute difference to the analytical Poiseuille profile divided by the **maximum** velocity (2.363 mm/s), not by the local velocity. It should not be read as a conventional relative error.

<p align="center">
<img src="results/paper_figures/Fig11_velocity_profile_and_mvnae.png" width="80%">
</p>

The scope is narrow on purpose: one flow condition, one particle size, one fluid and a straight channel. Behaviour at higher Reynolds number, dense seeding or other geometries is not inferred.

## IMPORTANT: what is in this repository

| Included | Not included |
|---|---|
| Full pipeline code written from the manuscript (`psv/`) | Raw experimental frames |
| CNN architecture exactly as in the paper | The 600 manually labelled patches |
| Every reported table and statistic as data, with scripts that re-derive them | Trained weights from the paper |
| A synthetic frame generator with ground truth, so the code runs end to end | The per-epoch log behind the published Fig. 6 |
| Real outputs of that synthetic demo (training log, confusion matrix, figures) | |
| Published paper figures | |
| JavaScript synthetic image tools | |

Results from the synthetic demo show that the code works. They are **not** the paper's numbers. Details are in [`docs/reproducibility_notes.md`](docs/reproducibility_notes.md), which also lists settings the manuscript leaves open (marked `[assumed]` in `psv/config.py`).

## Repository layout

```
.
├── psv/                          the Python package
│   ├── config.py                 every constant, tagged [paper] or [assumed]
│   ├── preprocessing.py          background, enhancement, threshold, contours, patches
│   ├── model.py                  Keras CNN (14,086,914 parameters)
│   ├── augment.py                rotation, flips, scaling (training set only)
│   ├── synthetic.py              synthetic frames with ground truth
│   ├── reconstruct.py            classify candidates, rebuild the denoised frame
│   ├── velocity.py               streak length, speed, Poiseuille reference, MVNAE
│   └── metrics.py                accuracy, precision, recall, F1, confusion matrix
├── scripts/
│   ├── 01_generate_synthetic_dataset.py
│   ├── 02_train_cnn.py           resumable, writes a genuine per-epoch history
│   ├── 03_evaluate_cnn.py
│   ├── 04_denoise_and_measure_demo.py
│   └── 05_reproduce_reported_results.py
├── synthetic_js/                 Node.js canvas tools that draw synthetic streak images
├── data/
│   ├── reported/                 tables and metadata taken from the manuscript
│   └── synthetic/                the demo dataset (about 4 MB, seed 7)
├── results/
│   ├── paper_figures/            Figs 1 to 11 as published
│   ├── reported_reproduction/    recomputed statistics, regenerated Fig. 11, Fig. 7 metrics
│   ├── demo_synthetic/           training history, test metrics, end-to-end figures
│   └── synthetic_js/             output of the JavaScript tools
├── docs/                         image processing, CNN, validation, setup, reproducibility
├── tests/test_psv.py             8 unit tests
├── run_demo.sh
├── requirements.txt
└── LICENSE
```

## Quick start

```bash
git clone <this-repository-url>
cd <this-repository>
pip install -r requirements.txt

python -m pytest -q tests                           # 8 fast tests
python -m psv.model                                 # prints the architecture, asserts 12,544 flatten size
python scripts/05_reproduce_reported_results.py     # re-derive the paper's statistics, redraw Fig. 11

./run_demo.sh 25                                    # full synthetic demo (about 10 min on one CPU core)
```

Training is checkpointed after every epoch. If a run is interrupted:

```bash
python scripts/02_train_cnn.py --epochs 25 --resume
```

JavaScript tools (optional, needs Node.js 18 or newer):

```bash
cd synthetic_js && npm install && npm run all
```

### Using the pipeline on your own frames

```python
import numpy as np, cv2
from psv import preprocessing as P
from psv.model import build_model
from psv.reconstruct import classify_candidates, reconstruct
from psv.velocity import streak_geometry, streak_speed_mms, signed_vector

refs = [cv2.imread(p, -1) for p in flow_off_frame_paths]       # 50 flow-off frames
B, SD = P.build_background(refs, return_std=True)

frame = cv2.imread("frame_0001.png", -1)
out = P.run_preprocessing(frame, B, SD)                         # all intermediates + candidates
model = build_model(); model.load_weights("my_weights.weights.h5")
pred, prob = classify_candidates(model, out["candidates"])
clean = reconstruct(out["subtracted"], out["candidates"], pred)

for c, keep in zip(out["candidates"], pred):
    if keep:
        L, angle, centre = streak_geometry(c.mask)
        v = signed_vector(angle, streak_speed_mms(L), flow_direction=(1, 0))
```

## CNN in one table

| Stage | Output |
|---|---|
| Input patch | 224 x 224 x 1 |
| 5 x [Conv 3x3 + ReLU + MaxPool 2x2] with 32, 64, 128, 256, 256 filters | 7 x 7 x 256 |
| Flatten | 12,544 |
| Dense 1024 + ReLU + Dropout 0.5 | 1024 |
| Dense 256 + ReLU + Dropout 0.5 | 256 |
| Dense 2 + Softmax (0 = replaced, 1 = retained) | 2 |

Adam (lr 0.001, beta1 0.9, beta2 0.999), 100 epochs in the paper. Augmentation on the training set only: rotation up to 30 degrees, flips, scaling by 10 %.

![CNN architecture](results/paper_figures/Fig03_cnn_architecture.png)

## What the synthetic demo shows

All numbers below come from `results/demo_synthetic/`, produced by the scripts in this repository (25 epochs, seed 7 for the data, seed 1 for training). The test set is 300 patches from frames never used for training.

| Measure | Synthetic demo |
|---|---|
| Train / validation / test patches | 253 / 47 / 300 |
| Test accuracy / precision / recall / F1 | 0.877 / 0.802 / 1.000 / 0.890 |
| Confusion matrix (true x predicted, replaced then retained) | [[113, 37], [0, 150]] |
| Artifact pixels removed on 20 unseen frames | 95.8 % |
| True streak pixels kept | 94.0 % |
| Measured vs true streak length | r = 0.997, mean bias -5.4 % |
| Depth-profile MVNAE (10 bins, normalised by maximum length) | mean 4.0 % |

All 37 errors are "replaced" patches predicted as "retained"; no valid streak was lost. In the patches I inspected, most are overlapping or adjacent streaks merged into one contour that looks like a single long streak, plus a few thick blurred streaks. The paper treats overlaps by labelling them "replaced", and this demo shows why that is the hard case. Do not compare these numbers with the 99.7 % from the real labelled dataset.

<p align="center">
<img src="results/demo_synthetic/end_to_end_example.png" width="85%">
</p>

## Documentation

| File | Content |
|---|---|
| [`docs/image_processing_pipeline.md`](docs/image_processing_pipeline.md) | Each preprocessing step, parameters, noise sources |
| [`docs/cnn_and_training.md`](docs/cnn_and_training.md) | Architecture, optimisation, dataset protocol, metrics |
| [`docs/velocity_reconstruction_and_validation.md`](docs/velocity_reconstruction_and_validation.md) | Poiseuille reference, MVNAE, Table 1 and 2, how to apply to new data |
| [`docs/experimental_setup.md`](docs/experimental_setup.md) | Channel, optics, camera, consistency checks |
| [`docs/reproducibility_notes.md`](docs/reproducibility_notes.md) | What can and cannot be reproduced, open items |

## Limitations

* Validated at a single flow condition (Re = 2.4) in a straight PDMS channel with 20 um particles in water.
* The percentile threshold favours brighter, better focused particles. The resulting bias on the velocity profile has not been quantified.
* One of 150 retained test candidates was rejected. Rejected candidates are not flagged in the reconstructed field.
* The CNN does not determine flow direction. Vector sign comes from the known bulk-flow direction.
* Overlapping streaks are classified as replaced, not separated.

## Citation

Please cite the paper once it is published. A `CITATION.cff` file is provided; add the DOI there when it is available.

## License

MIT, see [`LICENSE`](LICENSE).

## Contact

Open a GitHub Issue for questions about the code. Correspondence about the paper: Prof. Ramazan Asmatulu, Department of Mechanical Engineering, Wichita State University.

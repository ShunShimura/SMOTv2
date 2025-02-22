# Sliced Multi Object Tracker 

## Updates
- 25/02/02 Initial release.

## Performance
<p align="center">
  <img src="figures/meanIoUs.svg" width=100%>
  Solid lines indicate performance of proposed method while dased lines indicate ones of arranged existing methods.
</p>

## Installation
```
conda env create -f environment.yaml
conda activate SMOT
```

## Validation
Run validation on default settings.
```
python valid.py --ns 101 --nf 101
```
where `ns` means number of slice and `nf` means number of frames. 

If you use custom dataset, please run
```
python valid.py --ns {ns} --nf {nf} --input {your/dataset/root/path} --dv
```
where `dv` means scale ratio of state-space from observe-space.

And you can select the mode by 
```
python valid.py --ns {ns} --nf {nf} --mode {SD/K/KD}
```
- SD: Use 2D-like kalman filter to predict bboxes and leverage DepthSORT to get re-identified bboxes.
- K: Use only 3D-like kalman filter to predict and to estimate. (So don't use DepthSORT.)
- KD: Use 3D-like kalman filter to predict bboxes and leverage DepthSORT to get re-identified bboxes.

About other options, please see the function `parse_hyp()` on `valid.py`.

## Prediction
## Metrics

- Reconstruction:  MSE, LPIPS, SSIM, FID
- Component transfer: MSE$_{\text{irr}}$, IFG, FID
- Component editing:  FID

All command lines should be run in `Box-FaceS/metrics/`

**Reconstruction**

```bash
# First, reconstruct images and resave real images following Box-FaceS/boxfaces/reconstruction.py
# or Box-FaceS/vga/reconstruction.py
python eval_reconstruction.py --real output/reals --fake output/recon -bz 8 # returns MSE, LPIPS, SSIM
python -m pytorch_fid output/reals output/recon # returns FID
```

**Component transfer**

First, download the pre-trained StyleGAN2 discriminator from [stylegan2.pt]() and put it to `checkpoint/`

```bash
# First, reconstruct images following Box-FaceS/boxfaces/reconstruction.py
# Second, implement component transfer following Box-FaceS/boxfaces/comtrsf.py
python d_score.py --src output/recon --edit output/component-transfe/nose # returns MSE_irr, IFG
python -m pytorch_fid output/reals output/component-transfe/nose # returns FID
```

**Component editing**

```bash
# First, eiditing components following Box-FaceS/boxfaces/move.py
python -m pytorch_fid output/reals  output/move/eyes_up # for celeba-hq-dataset
# For vga dataset, editing objects following Box-FaceS/vga/move.py
python -m pytorch_fid output/fat/real  output/fat/move # for vga-dataset
```
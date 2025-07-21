## Box-FaceS — Official PyTorch Implementation

---

This repository contains the **official PyTorch implementation** of the paper:

**Box-FaceS: A Bidirectional Method for Box-Guided Face Component Editing**

> **Abstract:** *While the quality of face manipulation has been improved tremendously, the ability to control face components, e.g., eyebrows, is still limited. Although existing methods have realized component editing with user-provided geometry guidance, such as masks or sketches, their performance is largely dependent on the user's painting efforts. To address these issues, we propose Box-FaceS, a bidirectional method that can edit face components by simply translating and zooming the bounding boxes. This framework learns representations for every face component, independently, as well as a high-dimensional tensor capturing face outlines. To enable box-guided face editing, we develop a novel Box Adaptive Modulation (BAM) module for the generator, which first transforms component embeddings to style parameters and then modulates visual features inside a given box-like region on the face outlines. A cooperative learning scheme is proposed to impose independence between face outlines and component embeddings. As a result, it is flexible to determine the component style by its embedding, and to control its position and size by the provided bounding box. Box-FaceS also learns to transfer components between two faces while maintaining the consistency of image content. In particular, Box-FaceS can generate creative faces with reasonable exaggerations, requiring neither supervision nor complex spatial morphing operations. Through the comparisons with state-of-the-art methods, Box-FaceS shows its superiority in component editing, both qualitatively and quantitatively. To the best of our knowledge, Box-FaceS is the first approach that can freely edit the position and shape of the face components without editing the face masks or sketches.*

### [[PAPER (ACM MM 2022)]](https://dl.acm.org/doi/10.1145/3503161.3548392)
## Installation

---

Install the dependencies:
```bash
conda create -n boxfaces python=3.7
conda activate boxfaces
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```
For docker users:

```bash
docker pull huangwenjingcs/ubuntu18-conda-cuda11-pytorch1.7
```

## Preparing datasets for training

---

1. To obtain the CelebA-HQ dataset, please refer to the [Progressive GAN repository](https://github.com/tkarras/progressive_growing_of_gans). The official way of generating CelebA-HQ can be challenging. You can get the pre-generated dataset from [CelebA-HQ-dataset](https://drive.google.com/file/d/17wOT2Du1oKMU8DtRWupR_m1mgWvnGl1I/view?usp=sharing). Unzip the file and put the images them to "data/CelebA-HQ-img/".
2. To obtain the  Visual Genome Animals dataset, please refer to [VGA](https://drive.google.com/file/d/1WYBmk2pSBDtJcx-MFjJzwaX7FiDRL4sa/view?usp=sharing). Download the images and put them to "data/VGA-img/".  To obtain the full Visual Genome dataset, please refer to [VisualGenome](https://visualgenome.org/).
## Training networks

---

Once the datasets are set up, you can train the networks as follows:

1. Edit `configs/<DATASET>.json` to specify the dataset, model and training configurations.
1. Run the training script with `python train.py -c configs/<DATASET>.json `. For example, 
```bash
 # train Box-FaceS with CelebA-HQ dataset, with a batch size of 16
python train.py -c configs/celebahq.json --bz 16
 # train Box-FaceS with VGA dataset, with a batch size of 8 (paper setting on vga)
python train.py -c configs/vga.json --bz 8
```

   The code will use all GPUS by default, please specify the devices you want to use by:

```bash
 # train Box-FaceS in parallel, with a batch size of 16 (paper setting on celebahq)
CUDA_VISIBLE_DEVICES=0,1 python train.py -c configs/celebahq.json --bz 8
```

3. The checkpoints are written to a newly created directory `saved/models/<DATASET>`

Pre-trained models can be found on Google Drive:

| Path                                                         | Description                                              |
| ------------------------------------------------------------ | -------------------------------------------------------- |
| [checkpoint](https://drive.google.com/drive/folders/1gMhaWgNnE_0Ld-zOtVJ4kw4qxudW_Q0p?usp=sharing) | Main folder.                                             |
| ├  [celeba_hq_256.pt](https://drive.google.com/file/d/1RXluIY-MvDdbqS2dPxllQpCoywXFHQZV/view?usp=sharing) | Box-FaceS trained with CelebA-HQ dataset at 256×256.     |
| ├  [vga_256.pt](https://drive.google.com/file/d/19sO0xF4gr8cllW2uzOONgLvciB39Mx3H/view?usp=sharing) | Box-FaceS trained with Visual Genome Animals at 256×256. |

## Evaluation 

---

For **editing  the position and shape of component**, run:

```bash
python move.py --cmd <INSTRUCTION> # editing results are saved to output/move/<INSTRUCTION>
```

For **face component transfer**, run:

```bash
python replace.py --index <COMPONENT_INDEX> # editing results are saved to output/replace/
```

For **image reconstruction**, run:

```bash
python reconstruction.py # editing results are saved to output/recon
```

To reproduce the quantitative results in the paper:

```bash
# to reproduce Table 1
python move.py --cmd <INSTRUCTION>  --data_path data/test.txt --img_dir data/CelebA-HQ-img
# to reproduce Table 2
python comtrsf.py
# to reproduce Table 3
python reconstruction.py --data_path data/test.txt --img_dir data/CelebA-HQ-img
```


The results are saved to `output/`.

## Metrics 

---

- Reconstruction:  MSE, LPIPS, SSIM, FID
- Component transfer: MSE$_{\text{irr}}$, IFG, FID
- Component editing:  FID

If you want to see details, please follow `metrics/README.md`.

## Citation

---

If you find this work useful for your research, please cite our paper:
```
@inproceedings{huang2022box,
  title={Box-FaceS: A Bidirectional Method for Box-Guided Face Component Editing},
  author={Huang, Wenjing and Tu, Shikui and Xu, Lei},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={6061--6071},
  year={2022}
}
```

## Acknowledgement

---
This repository used some codes in [pytorch-template](https://github.com/victoresque/pytorch-template) and [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch).


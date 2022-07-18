## Evaluation 

---

For **editing  the position and shape of component**, run:

```bash
python move.py --cmd <INSTRUCTION> # editing results are saved to output/<INSTRUCTION>/move/
```

For **image reconstruction**, run:

```bash
python reconstruction.py # editing results are saved to output/recon
```

To reproduce the quantitative results in the paper:

```bash
# to reproduce Table 4
python reconstruction.py --data_path data/vga_test.txt --img_dir data/VGA-img
# to reproduce Table 5
python move.py --cmd <INSTRUCTION>  --data_path data/vga_test.txt --img_dir data/VGA-img
```


The results are saved to `output/`.

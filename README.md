# COGS

COGS: A Causal Representation Learning Framework for Out-of-Distribution Generalization in Time Series

### Building Conda Environment

```bash
cd COGS
conda env create -f environment.yml

conda activate cogs
```

### Running

Prepare your own data and move it to ```./data``` dir. Here we provide an example of DSADS dataset and you can download it from [DSADS Dataset](https://ieee-dataport.org/documents/daily-and-sports-activities-data-set).

After data preparation, you can run the following script.

```bash
python main.py
```


### Citations
```
@inproceedings{song2026cogs,
  title={COGS: A Causal Representation Learning Framework for Out-of-Distribution Generalization in Time Series},
  author={Song, Xinxin and Cheng, Yuxiao and Xiao, Tingxiong and Suo, Jinli},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={30},
  pages={25572--25580},
  year={2026}
}
```

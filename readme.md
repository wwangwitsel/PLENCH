# PLENCH
This repository is the official implementation of the paper "Realistic Evaluation of Deep Partial-Label Learning Algorithms" and technical details of this approach can be found in the paper. 

## Requirements
- Python 3.6.13
- numpy 1.19.2
- Pytorch 1.7.1
- torchvision 0.8.2
- pandas 1.1.5
- scipy 1.5.4
- tqdm
- Pillow

## Dataset
Most tabular datasets can be found at https://palm.seu.edu.cn/zhangml/Resources.htm#data. PLCIFAR10 can be found at 

### Run an Algorithm
```
python -m plench.train --data_dir=<your dataset path> --algorithm PRODEN --dataset PLCIFAR10_Aggregate  --output_dir=<your output path> --steps 60000 --skip_model_save --checkpoint_freq 1000
```

## Run Algorithms in Batch
```
python -m plench.sweep launch --data_dir=<your dataset path> --command_launcher multi_gpu --n_hparams_from 0 --n_hparams 20 --n_trials_from 0 --n_trials 3 --datasets PLCIFAR10_Aggregate PLCIFAR10_Vaguest --algorithms PRODEN CAVL --output_dir=<your output path> --skip_confirmation --skip_model_save --steps 60000
```

## Collect Experimental Results 
```
python -m plench.collect_results --input_dir=<path of the output files>
```

## Acknowledgement
The code was based on the codebase of the following paper:

- Ishaan Gulrajani and David Lopez-Paz. In search of lost domain generalization. In Proceedings of the 9th International Conference on Learning Representations, 2021.


## ## Citation
```
@inproceedings{wang2025realistic,
    author = {Wang, Wei and Wu, Dong-Dong and Wang, Jindong and Niu, Gang and Zhang, Min-Ling and Sugiyama, Masashi},
    title = {Realistic evaluation of deep partial-label learning algorithms},
    booktitle = {Proceedings of the 13th International Conference on Learning Representations},
    year = {2025}
}
```



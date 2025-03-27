## SLRed : A data-driven sparse learning approach to reduce chemical reaction mechanisms

Kinetic mechanism reduction method using sparse lesrning (SL) to identify influential reactions proposed by **Spray & Combustion Laboratory** in School of Astronautics in Beihang University (BUAA). Codes for generating datasets, optimization of sparse weight vector and iterative evaluation of reduced mechansims with SL method are provided. A detailed introduction to SL method can be found at [Fang2024].

Existing mechanism reduction methods including DRGEP, DRGEPSA and detailed reduction (DR)[Wang1991] are also provided. DRG and DRGEPSA methods are adopted from $\texttt{pyMARS}$ with modifications to fit $\texttt{Cantera}$ 3.0. DR method is implemented based on the instructions of [Liu2021].

### How to Run
To perform mechanism reduction using SL method, the detailed mechanism should be firstly uploaded to the ```./mechs``` directory following the format of ```yaml``` file for $\texttt{Cantera}$ 3.0 and the configuration file should be uploaded to ```./cfgs``` directory where templates are provided there. SL method can be run with:
```
python main.py --method sl --fuel xxx
```
Paramaters for SL method can be modified in the configuration files. Additionally, DRGEP or DRGEPSA methods can be run with:
```
python main.py --method drgep(sa) --fuel xxx
```
The difference between configuration files for the two method is referred to $\texttt{pyMARS}$ docs. DR method can be run with
```
python main.py --method dr --fuel xxx
```

### Requirements
```
absl
ml_collections
h5py
numpy
cantera >= 3.0.0
torch
einops
ruamel
```

### How to Contribute
$\texttt{SLRed}$ is a research program rather than a fully tested software package. The efficiency and correctness of cases, or input values that differ greatly from the default or published values are not guaranteed. We welcome help with extending the capabilities of $\texttt{SLRed}$. If interested, please contact Shen Fang eureka10shen@buaa.edu.cn, Wang Han drwanghan@buaa.edu.cn and Lijun Yang yanglijun@buaa.edu.cn, or a member of the research group.

### Citation
```
@article{fang2024data,
  title={A data-driven sparse learning approach to reduce chemical reaction mechanisms},
  author={Fang, Shen and Zhang, Siyi and Li, Zeyu and Fu, Qingfei and Zhou, Chong-Wen and Han, Wang and Yang, Lijun},
  journal={arXiv preprint arXiv:2410.09901},
  year={2024}
}
```

### References
[Fang2024] S. Fang, S. Zhang, Z. Li, et al. A data-driven sparse learning approach to reduce chemical reaction mechanisms, https://arxiv.org/abs/2410.09901 (2024) \
[Wang1991] H. Wang and M. Frenklach, Detailed Reduction of Reaction Mechanisms for Flame
Modeling, *Combustion and Flame* 87, 365-370 (1991). \
[pyMARS] P. Mestas, P. Clayton, and K. Niemeyer, https://github.com/Niemeyer-Research-Group/pyMARS (2018) \
[Liu2021] Z. Liu, J. Oreluk, A. Hegde, et al. Does a reduced model reproduce the uncertainty of the original full-size model? *Combustion and Flame* 226, 98-107 (2021)

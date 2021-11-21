# Code supplementary for the CILP paper

CILP: Co-simulation based Imitation Learner for Dynamic Resource Provisioning in Cloud Computing. IJCAI - 2022.


## CILP Approach
We present a novel VM provisioner CILP in this paper that uses a Transformer based co-simulated imitation learner.

## Quick Start Guide
To run the code, install required packages using
```bash
pip3 install matplotlib scikit-learn
pip3 install -r requirements.txt
pip3 install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

To run the code with the required provisioner and dataset use the following
```bash
python3 main.py --provisioner <provisioner> --workload <workload>
```
Here `<provisioner>` can be one of `ACOARIMA`, `ACOLSTM`, `DecisionNN`, `SemiDirect`, `UAHS`, `Narya`, `CAHS`, `CILP_IL`, `CILP_Trans` and `CILP`. Also, `<workload>` can be one of `Azure2017`, `Azure2019` and `Bitbrain`.

Sample command:
```bash
python3 main.py --provisioner CILP --workload Azure2017
```
To run the code with the required scheduler, modify line 104 of main.py to one of the several options including LRMMTR, RF, RL, RM, Random, RLRMMTR, TMCR, TMMR, TMMTR, GA, GOBI.

# CILP

CILP: Co-simulation based Imitation Learner for Resource Provisioning in Cloud Computing.
IJCAI - 2022.

Prior work does not consider energy, provisioning and migration overheads. (1) Energy: the cpu to power consumption mapping for a cloud node is considered to be stable with time and this has been used in many simulated environments. (2) Provisioning: the time taken to provision cloud nodes is consistent and the conditional distribution of cost conditioned on resource type is stationary over time. (3) Migration: Overheads

Here, (2) and (3) affect utilization ratio. Also, we consider deployment cost in our model and provisioning is not free. QoS is considered as a convex combination of energy, sla violation rate and cost (HUNTER). The weightage between utilization ratio and cost can be set as per user. Our method is agnostic to deployment platform. For public cloud cost important, for private cloud utilization ratio is important.

## Figures

* Fig: Neural Model
* Fig: Interaction between co-simulator and IL (top-level design)
* Alg: CILP algo
* @Tab: Results and ablation (r, cost, qos)
* @Tab: Results with gamma hyperparameter (r, cost, qos)
* @Fig: QoS results (energy pi, response time pi, sla pi, provisioning overhead, migrations overhead)

## License

BSD-3-Clause. 
Copyright (c) 2021, Shreshth Tuli.
All rights reserved.

See License file for more details.

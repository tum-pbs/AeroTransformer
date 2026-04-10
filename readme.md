# AeroTransformer: Towards a foundation-model style aerodynamic surrogate model

This repository contains the open-sourced resources for the AeroTransformer described in:

**Towards a Foundation-Model Paradigm for Aerodynamic Prediction in Three-dimensional Shape Design**  
Yunjia Yang, Babak Gholami, Caglar Gurbuz, Mohammad Rashed, Nils Thuerey

The project follows a two-stage workflow:

1. **Pre-training** on a large and diverse wing dataset (SuperWing).
2. **Fine-tuning** on a task-specific local design space (e.g., CRM-perturbed wings).

The goal is to build reusable aerodynamic surrogate models that remain accurate under limited task-specific data.

## Resources Overview
- Paper

- Dataset collection:  
  - SuperWing (Pre-training dataset)        
    [https://huggingface.co/datasets/yunplus/SuperWing](https://huggingface.co/datasets/yunplus/SuperWing)
  - CRMpert (Task-specfic fine-tuning dataset)
  [https://huggingface.co/datasets/thuerey-group/CRMpert](https://huggingface.co/datasets/thuerey-group/CRMpert)
- Model hyperparameter collection (Pre-trained and fine-tuned):  
  [https://huggingface.co/thuerey-group/AeroTransformer](https://huggingface.co/thuerey-group/AeroTransformer)
- Implementation dependency:  
  - Model implementation (`FloGen` repo)
    [https://github.com/YangYunjia/floGen](https://github.com/YangYunjia/floGen)
  - Wing postprocess and visualazation (`cfdpost` repo) 
    [https://github.com/YangYunjia/cfdpost](https://github.com/YangYunjia/cfdpost)
- Training and simulation source code (here)
- `WebWing` Online interactive wing design tool:
    - Online version: [https://webwing.pbs.cit.tum.de/](https://webwing.pbs.cit.tum.de/)
    - Source code: [https://github.com/YangYunjia/webwing](https://github.com/YangYunjia/webwing)



## Repository Scope
 
This repo is intentionally focused on **training source code** and **simulation-related scripts** for the AeroTransformer project.


```text
AeroTransformer/
├── training/
│   ├── pretrain.py      # pre-train AeroTransformer
│   ├── finetune.py      # fine-tune a pretrained model on downstream task data
│   └── postprocess.py   # evaluate field/coefficient errors
├── simulation/
│   ├── gen-mesh.crmpert.py      # surface + volumetric mesh generation for CRMpert dataset
│   ├── gen-mesh.superwing.py    # surface + volumetric mesh generation for SuperWing dataset
│   ├── original_tip.xyz  # tempelate wing tip shape and mesh
│   ├── run-adflow.py     # calling the pyADflow solver
│   └── single-point.py   # main function for simulating single/multiple samples for one shape
├── LICENSE
└── readme.md
```

## Environment Setup

> Simulation part requires [MDO Lab](https://mdolab-mach-aero.readthedocs-hosted.com/en/latest/installInstructions/dockerInstructions.html) docker image (at least with `ADflow`, `pyHyp`, `cgnsutilities`)


## Citation


## License

This repository is released under the license provided in `LICENSE`.

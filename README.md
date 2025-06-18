# Source codes for FedWSIDD
> We provide bash script for training on CAMELYON16 and CAMELYON17 dataset
> Before running the code, ensure you have downloaded these dataset and preprocess following [CLAM](https://github.com/mahmoodlab/CLAM)

The list of useful parameters is as follows:
* `data_root_dir`: this is the path where you put your '.pt' files for patch features.
* `feature_type`: feature extractor, e.g., ResNet50, ViT, UNI, PLIP, CONCH.
* `mil_method`: MIL framework, e.g., CLAM, ABMIL, TransMIL.
* `fed_method`: Federated Learning frameworks, e.g., FedAvg, FedProto, MOON.
* `heter_model`: set this to True to allow Heterogeneous Local Model setup.
* `ipc`: number of synthetic slides per class.
* `nps`: number of synthetic patches per slide.
* `syn_size`: size of synthetic patches.
* `init_real`: set this to True to allow initialising synthetic patches with real patches.

### Run (CAMELYON16 Example)
```bash
bash run.sh
```

import random
import torch
import numpy as np
import Trans_mod
import hydra
from omegaconf import DictConfig
import math
import logging

log = logging.getLogger(__name__)

@hydra.main(version_base=None,config_path="config",config_name="samson")
def main(cfg:DictConfig):
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(cfg)
    # Device Configuration
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("\nSelected device:", device, end="\n\n")

    tmod = Trans_mod.Train_test(dataset='samson_K_3', device=device, skip_train=False, save=True,cfg=cfg)
    final_result = tmod.run(smry=False)
    print(final_result)
    if(math.isnan(final_result)):
        final_result = 100.0 # a very large error for this problem
    else:
        final_result = float(final_result)
    log.info(final_result)
    return final_result



if __name__ == "__main__":
    main()

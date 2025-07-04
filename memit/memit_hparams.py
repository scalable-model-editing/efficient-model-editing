from dataclasses import dataclass
from typing import List, Literal

from util.hparams import HyperParams


@dataclass
class MEMITHyperParams(HyperParams):
    # Method
    layers: List[int]
    layer_selection: Literal["all", "random"]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    mom2_update_weight: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str

    #Objective
    calculate_objective_value: bool
    update_norm_lambda: float
    emmet_lambda: float

    # Additonal parameters for fastEMMET
    dynamic_multiplier: float
    dynamic: bool
    shuffle_text: bool
    wikipedia_lang: str
    skill_preservation: bool
    skill_lambda: float
    skill_sample_size: int
    random_vector_preservation: bool
    proj_dim: int
    identity_cov: bool

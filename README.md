# Efficient Knowledge Editing with Minimal Pre-computation

# A Unified & Efficient Framework for Model Editing

Based on Unified Model Editing Framework, our FastMEMIT family of methods add a "dynamic multiplier" hyperparameter to control and reduce the number of preserved key vectors used in the pre-computation. 

Our FastMEMIT methods can not only finish the pre-computation step with less than 0.1% of the original stipulated number of hidden vectors, which reduces the time from tens of hours to a few minutes, but also achieve similar or improved efficacy, paraphrase, and neighborhood scores compared to the original algorithms. 

## Performance Visualization
Here is the plot comparison between FastEMMET and EMMET, and FastMEMIT and MEMIT in Llama 2:

<div align="center">
  <img width="588" alt="Performance of FastEMMET and FastMEMIT in Llama 2 across different batch sizes, showing Overall, Efficacy, Paraphrase, and Neighborhood Scores." src="https://github.com/user-attachments/assets/d6c4a150-5204-4d80-a0da-1eb37d156925" />
</div>

Here are precise stats of dynamic multipliers=10 in tables indicating the comparison between FastEMMET and EMMET, and FastMEMIT and MEMIT in Llama 2:

<div align="center">
  <img width="588" alt="Detailed comparison table for EMMET and FastEMMET metrics in Llama 2 at dynamic multiplier=1, showing Efficacy, Generalization, Locality, and Overall Score." src="https://github.com/user-attachments/assets/5ff52e86-dd67-403f-95c5-dd348cf93e95" />
</div>

<div align="center">
  <img width="588" alt="Detailed comparison table for MEMIT and FastMEMIT metrics in Llama 2 at dynamic multiplier=1, showing Efficacy, Generalization, Locality, and Overall Score." src="https://github.com/user-attachments/assets/2da9b023-29dd-42c3-8735-e2aa3bab50e7" />
</div>

## Installation
We work off of the [MEMIT](https://github.com/kmeng01/memit) codebase, so we'll reference the same installation procedures here: 
"We recommend `conda` for managing Python, CUDA, and PyTorch; `pip` is for everything else. To get started, simply install `conda` and run:
```bash
CONDA_HOME=$CONDA_HOME ./scripts/setup_conda.sh
```

`$CONDA_HOME` should be the path to your `conda` installation, e.g., `~/miniconda3`."


## Running the experiments

To evaluate EMMET with FastMEMIT family of Methods, 

you need to set "dynamic=true" first, and change "dynamic_multiplier" to reduce preserved key vectors in hparams files.

(Note: "dynamic_multiplier=10" means using 10 times the theoretical minimum number of preserved key vectors for pre-computation.)

Then run the following command:

```python
python experiments/evaluate_unified_editing.py \
--alg_name=EMMET \
--num_edits=4 \
--model_name=gpt2-xl \
--hparams_fname=gpt2-xl.json \
--ds_name=cf \
--sequential=True
```

The above script can also be used to run ROME and MEMIT from the same file. We have a common underlying code-base for calculating the key and value vectors.

**Before any experiment is run**, there might be need to update ```sys.path.append('/path/to/unified-model-editing')``` in the files 'experiments/evaluate_unified_editing.py' and 'experiments/py/eval_utils_zsre.py' 

## How to Cite
If you find our work useful, please cite it using the following:


```bibtex
@article{gupta2024unified,
  title={A Unified Framework for Model Editing},
  author={Gupta, Akshat and Sajnani, Dev and Anumanchipalli, Gopala},
  journal={arXiv preprint arXiv:2403.14236},
  year={2024}
}
```

```bibtex
@article{gupta2024model,
  title={Model Editing at Scale leads to Gradual and Catastrophic Forgetting},
  author={Gupta, Akshat and Rao, Anurag and Anumanchipalli, Gopala},
  journal={arXiv preprint arXiv:2401.07453},
  year={2024}
}
```

# FastMEMIT Family of Methods: Efficient Knowledge Editing with Minimal Pre-computation

# A Unified & Efficient Framework for Model Editing

Based on Unified Model Editing Framework, our FastMEMIT family of methods add a "dynamic multiplier" hyperparameter to control and reduce the number of preserved key vectors used in the pre-computation. 

Our FastMEMIT methods can not only finish the pre-computation step with less than 0.1% of the original stipulated number of hidden vectors, which reduces the time from tens of hours to a few minutes, but also achieve similar or improved efficacy, paraphrase, and neighborhood scores compared to the original algorithms. 

## Installation
We work off of the [MEMIT](https://github.com/kmeng01/memit) codebase, so we'll reference the same installation procedures here: 
"We recommend `conda` for managing Python, CUDA, and PyTorch; `pip` is for everything else. To get started, simply install `conda` and run:
```bash
CONDA_HOME=$CONDA_HOME ./scripts/setup_conda.sh
```

`$CONDA_HOME` should be the path to your `conda` installation, e.g., `~/miniconda3`."


## Running the experiments

To evaluate EMMET with FastMEMIT family of Methods, 

you need to set "dynamic=true" first, and change "dynamic_multiplier" to reduce preserved key vectors.

(Note: "dynamic_multiplier=10" means using 10 times the theoretical minimum number of preserved key vectors for pre-computation.)

Then run the following command:

```python
python experiments/evaluate_unified_editing.py \
--alg_name=EMMET \
--num_edits=4 \
--model_name=gpt2-xl \
--hparams_fname=gpt2-xl.json \
--ds_name=cf
```

The above script can also be used to run ROME and MEMIT from the same file. We have a common underlying code-base for calculating the key and value vectors.

**Before any experiment is run**, there might be need to update ```sys.path.append('/path/to/unified-model-editing')``` in the files 'experiments/evaluate_unified_editing.py' and 'experiments/py/eval_utils_zsre.py' 

## Downstream Evaluation

**downstream_tasks** specifies the downstream tasks to run. Available tasks: nli,rte,mrpc,sentiment_analysis,dialogue,nli,cola,sst

**number_of_few_shots** is the number of few shots for each downstream task. Specify the number of few shots for each task, separated by commas. number_of_few_shots must be same length as downstream_tasks. Its default value is 0 when the flag is not provided

**number_of_tests** is the number of tests for all downstream tasks. The default to using the entire test dataset if the flag is not provided

Example:
To run nli, sst and mmlu with 2,3,3 few shots respectively, run the following command:

```python
python experiments/evaluate_unified_editing.py \
--alg_name=EMMET \
--num_edits=4 \
--model_name=gpt2-xl \
--hparams_fname=gpt2-xl.json \
--ds_name=cf \
--do_downstream_eval=True \
--downstream_eval_steps=20 \
--downstream_tasks=nli,sst,mmlu \
--number_of_few_shots=2,3,3 \
--number_of_tests=20
```

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

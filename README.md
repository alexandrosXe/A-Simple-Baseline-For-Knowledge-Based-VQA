# A-Simple-Baseline-For-Knowledge-Based-VQA
Repo for the EMNLP 2023 paper "A Simple Knowledge-Based Visual Question Answering"

## Abstract
This paper is on the problem of KnowledgeBased Visual Question Answering (KB-VQA).
Recent works have emphasized the significance
of incorporating both explicit (through external
databases) and implicit (through LLMs) knowledge to answer questions requiring external
knowledge effectively. A common limitation
of such approaches is that they consist of relatively complicated pipelines and often heavily
rely on accessing GPT-3 API. Our main contribution in this paper is to propose a much simpler and readily reproducible pipeline which,
in a nutshell, is based on efficient in-context
learning by prompting LLaMA (1 and 2) using
question-informative captions as contextual information. Contrary to recent approaches, our
method is training-free, does not require access
to external databases or APIs, and yet achieves
state-of-the-art accuracy on the OK-VQA and
A-OK-VQA datasets. Finally, we perform several ablation studies to understand important
aspects of our method. Our code is publicly
available at https://github.com/alexandrosXe/ASimple-Baseline-For-Knowledge-Based-VQA

## Install
First, please install the necessary dependencies:
```bash
pip install -r requirements.txt
```
## Usage
First, download the LLaMa weights and convert them to Huggingface format:
* Weights for the LLaMA models can be obtained from by filling out this [form](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform)
* After downloading the weights, they will need to be converted to the Hugging Face Transformers format using the conversion [script](https://huggingface.co/docs/transformers/main/model_doc/llama).   


## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@misc{xenos2023simple,
      title={A Simple Baseline for Knowledge-Based Visual Question Answering}, 
      author={Alexandros Xenos and Themos Stafylakis and Ioannis Patras and Georgios Tzimiropoulos},
      year={2023},
      eprint={2310.13570},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Contact

**Please feel free to get in touch at**: ` a.xenos@qmul.ac.uk`

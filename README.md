# A-Simple-Baseline-For-Knowledge-Based-VQA
Repo for the EMNLP 2023 paper "A Simple Knowledge-Based Visual Question Answering"

### Abstract
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

### Install
First, please install the necessary dependencies:
pip install -r requirements.txt

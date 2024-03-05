# Paired Bio-Inspired Antibody Language Model (BALM)

This repository contains the code for fine-tuning the BALM model for natively paired antibody sequences.

## Create Environment

To set up the environment, run the following commands:

```bash
conda env create -f environment.yml
conda activate PBALM
```


## Preparation
The pre-trained weights of BALM should be downloaded from Google Drive link: [pretrained-BALM](https://drive.google.com/drive/folders/1foy264CIawBIT3QFTdc6JBVxw6MQfvQd?usp=sharing). Place the downloaded files in the `pretrained_BALM` folder.

### Data Preprocessing

To download, cluster, and clean the data from Paired OAS dataset and calculate the mask probabilities, you can simply run:
```bash
bash data.sh
```
This script will create two files in the main directory: data.pkl and mask_probs.pt. These files contain pickled antibody sequences with their IMGT numbering and the probability of masking for each IMGT position calculated based on the BALM paper, respectively.
## Run inference

```
from BALM.modeling_balm import BALMForMaskedLM
from numbering import get_anarci_numbering
from transformers import EsmTokenizer
import torch

# an antibody sequence example
light = "DIQMTQSPSSLSASVGDRVTITCRASQGIRNDLGWYQQKPGKAPKRLIYAASSLQSGVPSRFSGSGSGTEFTLTISSLQPEDFATYYCLQHNSYPRTFGQGTKVEIK"
heavy = "EVQLLESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDWPFWQWLVRRGERFDYWGQGTLVTVSS"
input_ab = [[light, heavy]]

tokenizer = EsmTokenizer.from_pretrained("BALM/tokenizer/vocab.txt", do_lower_case=False, model_max_length=288)

batch = tokenizer(input_ab, truncation=True, padding="max_length", return_tensors="pt")
# generate position_ids
batch.update(get_anarci_numbering(input_ab[0]))

with torch.no_grad():
    # please download from Google drive link before
    model = BALMForMaskedLM.from_pretrained("./pretrained_PBALM/")
    # on CPU device
    outputs = model(**batch, return_dict=True, output_hidden_states=True, output_attentions=True)

    # final hidden layer representation [batch_sz * max_length * hidden_size]
    final_hidden_layer = outputs.hidden_states[-1]
    
    # final hidden layer sequence representation [batch_sz * hidden_size]
    final_seq_embedding = final_hidden_layer[:, 0, :]
    
    # final layer attention map [batch_sz * num_head * max_length * max_length]
    final_attention_map = outputs.attentions[-1]
```


<!-- ## Citation
If you find our model is useful for you, please cite as:

```
@article{jing2023accurate,
  title={Accurate Prediction of Antibody Function and Structure Using Bio-Inspired Antibody Language Model},
  author={Jing, Hongtai and Gao, Zhengtao and Xu, Sheng and Shen, Tao and Peng, Zhangzhi and He, Shwai and You, Tao and Ye, Shuang and Lin, Wei and Sun, Siqi},
  journal={bioRxiv},
  pages={2023--08},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
``` -->

The architecture and finetuning process of Paired-BALM builds on the BALM and Hugging Face modeling framework. We really appreciate the work of [BALM](https://github.com/BEAM-Labs/BALM) and [Hugging Face](https://huggingface.co/) team.


## License
This source code is licensed under the MIT license found in the `LICENSE` file in the root directory of this source tree.

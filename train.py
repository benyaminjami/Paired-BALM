from datasets import Dataset
import pickle
from BALM.modeling_balm import BALMForMaskedLM
from transformers import EsmTokenizer
import torch
import re
import os
import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, List, Dict
from transformers import PreTrainedTokenizerBase, TrainingArguments, Trainer
from transformers.data.data_collator import _torch_collate_batch, DataCollatorMixin
import transformers

transformers.set_seed(42)


def get_numbering(anarci_numbering, max_len=288):
    """
    Convert anarci numbering to position and chain IDs.

    Args:
        anarci_numbering (list): List of tuples containing the anarci numbering.
        max_len (int, optional): Maximum length of the position IDs. Defaults to 288.

    Returns:
        dict: A dictionary containing the position IDs and chain IDs as torch tensors.
    """
    index = []
    chain = []
    # ignore "-"

    for element in anarci_numbering:
        indexes = element[0]
        c = ([0] * (len(element[0]) + 2)) + ([1] * (len(element[1]) + 1))
        chain.append(c + ([0] * (max_len - len(c))))
        indexes.append("140")
        indexes.extend(element[1])
        try:
            new_index = list(map(lambda id: int(id), indexes))
        except:
            new_index = []
            for id in indexes:
                try:
                    new_index.append(int(id))
                except:
                    pos_map = {
                        "111A": 129,
                        "111B": 130,
                        "111C": 131,
                        "111D": 132,
                        "111E": 133,
                        "112A": 139,
                        "112B": 138,
                        "112C": 137,
                        "112D": 136,
                        "112E": 135,
                        "112F": 134,
                    }
                    if (
                        id not in pos_map.keys()
                        and int(re.sub("[a-zA-Z]", "", id)) < 111
                    ):
                        new_index.append(int(re.sub("[a-zA-Z]", "", id)))
                    elif id in pos_map.keys():
                        new_index.append(pos_map[id])
                    elif int(id[:3]) == 111:
                        new_index.append(133)
                    elif int(id[:3]) == 112:
                        new_index.append(134)

        new_index = [0] + new_index + [140] * (max_len - 1 - len(new_index))
        if len(new_index) > max_len:
            new_index = new_index[:max_len]
            new_index[-1] = 140
        index.append(new_index)
    return {"position_ids": torch.tensor(index), "chain_ids": torch.tensor(chain)}


@dataclass
class DataCollatorForGuidedLanguageModeling(DataCollatorMixin):
    """
    Data collator for guided language modeling.

    This class handles the collation of input examples and the preparation of masked tokens inputs/labels
    for masked language modeling.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer used for tokenization and padding.
        mask_probs (torch.FloatTensor): The probability matrix for masking tokens.
        mlm (bool, optional): Whether to perform masked language modeling. Defaults to True.
        mlm_probability (float, optional): The probability of masking a token during masked language modeling.
            Defaults to 0.15.
        pad_to_multiple_of (Optional[int], optional): The padding length for input sequences. Defaults to None.
        tf_experimental_compile (bool, optional): Whether to use TensorFlow experimental compile mode.
            Defaults to False.
        return_tensors (str, optional): The type of tensors to return. Defaults to "pt".
    """

    tokenizer: PreTrainedTokenizerBase
    mask_probs: torch.FloatTensor
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Collate input examples and prepare masked tokens inputs/labels for masked language modeling.

        Args:
            examples (List[Union[List[int], Any, Dict[str, Any]]]): The input examples to collate.

        Returns:
            Dict[str, Any]: The collated batch with input_ids and labels.
        """
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(
                examples,
                return_tensors="pt",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        else:
            batch = {
                "input_ids": _torch_collate_batch(
                    examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
                )
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], batch, special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_mask_tokens(
        self, inputs: Any, batch: Any, special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.

        Args:
            inputs (Any): The input tokens.
            batch (Any): The batch of input examples.
            special_tokens_mask (Optional[Any], optional): The mask for special tokens. Defaults to None.

        Returns:
            Tuple[Any, Any]: The masked inputs and labels.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = self.mask_probs[batch["chain_ids"], batch["position_ids"]]
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


def train(args):
    with open("data.pkl", "rb") as f:
        d = pickle.load(f)
    print('Number of samples: ', len(d['seqs']))
    model = BALMForMaskedLM.from_pretrained(args.pretrain_dir)
    ds = Dataset.from_dict(d).shuffle()
    ds = ds.train_test_split(test_size=0.05)

    tokenizer = EsmTokenizer.from_pretrained(
        "BALM/tokenizer/vocab.txt", do_lower_case=False, model_max_length=288
    )

    def preprocess_function(samples):
        s = tokenizer(
            samples["seqs"], truncation=True, padding="max_length", return_tensors="pt"
        )
        p = get_numbering(samples["pos"])
        s.update(p)
        return s

    ds = ds.map(
        preprocess_function,
        batched=True,
        num_proc=8,
    )
    mask_probs = torch.stack(torch.load("mask_probs.pt"))
    data_collator = DataCollatorForGuidedLanguageModeling(tokenizer, mask_probs)

    os.environ["WANDB_PROJECT"] = "balm_paired"  # name your W&B project

    training_args = TrainingArguments(
        fp16=True,
        evaluation_strategy="steps",
        save_strategy="epoch",
        seed=42,
        per_device_train_batch_size=52,
        per_device_eval_batch_size=52,
        logging_steps=100,
        eval_steps=500,
        adam_beta1=0.9,
        adam_beta2=0.999,
        weight_decay=0.01,
        warmup_steps=2000,
        learning_rate=2e-5,
        gradient_accumulation_steps=1,
        run_name="paired_balm",
        output_dir=args.out_dir,
        num_train_epochs=15,
        logging_dir="logs/",
        report_to="wandb",
        logging_first_step=True,
        lr_scheduler_type="inverse_sqrt",
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=data_collator,
        # callbacks=transformers.EarlyStoppingCallback(early_stopping_patience=3,)
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrain-dir",
        type=str,
        default='pretrained_BALM',
        help="folder to pretrained files of BALM",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default='checkpoints',
        help="destination for checkpoints",
    )

    train(parser.parse_args())

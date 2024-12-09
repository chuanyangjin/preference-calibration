"""
python rm_training_with_disagreement.py \
    --model_name_or_path=facebook/opt-350m \
    --output_dir="reward_modeling_anthropic_hh" \
    --per_device_train_batch_size=64 \
    --num_train_epochs=1 \
    --gradient_accumulation_steps=16 \
    --gradient_checkpointing=True \
    --learning_rate=1.41e-5 \
    --report_to="wandb" \
    --remove_unused_columns=False \
    --optim="adamw_torch" \
    --logging_steps=10 \
    --evaluation_strategy="steps" \
    --max_length=512 \
"""

import warnings
import random
import torch
from datasets import load_dataset, Dataset, DatasetDict
from datasets import concatenate_datasets
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser

from trl import ModelConfig, RewardConfig
from trl import RewardTrainer
from trl import get_kbit_device_map, get_peft_config, get_quantization_config


tqdm.pandas()

def rankings_to_margins(selected, rankings):
    assert selected == 0 or selected == 1
    assert type(rankings) == list
    if len(rankings) == 0
        return None
    # margin = 1 if selected option is much better than the other one, 0 if both options are very similar
    # So margin = 0 if avg(rankings) = 0.5, and margin = 1 if avg(rankings) = 0 or 1
    half_score = len(rankings) / 2 
    margin = abs(sum(rankings) - half_score))/half_score
    return margin


if __name__ == "__main__":
    parser = HfArgumentParser((RewardConfig, ModelConfig))
    reward_config, model_config = parser.parse_args_into_dataclasses()
    # reward_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ################
    # Model & Tokenizer
    ################
    # torch_dtype = (
    #     model_config.torch_dtype
    #     if model_config.torch_dtype in ["auto", None]
    #     else getattr(torch, model_config.torch_dtype)
    # )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True, cache_dir = "/scratch/vp1271/hf_cache", token="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")
    if "llama" in model_config.model_name_or_path:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    # tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
            model_config.model_name_or_path, num_labels=1, cache_dir = "/scratch/vp1271/hf_cache", token="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", **model_kwargs
    )

    if model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script."
        )

    ################
    # Dataset
    ################
    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(chosen)
            tokenized_rejected = tokenizer(rejected)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples
    
    def preprocess_function_webgpt(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for answer_0, score_0, answer_1, score_1 in zip(examples["answer_0"], examples["score_0"], examples["answer_1"], examples["score_1"]):
            if score_0 > score_1:
                chosen, rejected = answer_0, answer_1
            else:
                rejected, chosen = answer_1, answer_0
            tokenized_chosen = tokenizer(chosen)
            tokenized_rejected = tokenizer(rejected)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

    def convert_summarize_to_pairwise(examples):
        pairs = []
        prev_id = None
        for i, example in enumerate(examples):
            if type(example['info']['post']) != str:
                continue
            question = example['info']['post'] + '\n\nPlease summarize the text above for me.'
            answers = [example['summaries'][0]['text'], example['summaries'][1]['text']]
            selected_id = example['choice']
            other_id = (selected_id + 1)%2
            assert selected_id + other_id == 1
            pairs.append({'chosen':question+answers[selected_id], 'rejected':question+answers[other_id]})
        print(len(examples), len(pairs))
        return pairs
        """
            if examples[i]['info']['id'] == prev_id:
                continue
            id_start = i
            while examples[i]['info']['id'] == examples[i+1]['info']['id']:
                i += 1
            id_end = i
            prev_id = examples[i]['info']['id']
            if id_end - id_start >= 1:
                ids = random.sample(range(id_start, id_end + 1), 5)
                assert (examples[ids[0]]['info']['post'] == examples[ids[1]]['info']['post'])
                for random_id in ids:
                    question = examples[random_id]['info']['post'] + '\n\nPlease summarize the text above for me.'
                    answer_0 = examples[random_id]['summaries'][0]['text']
                    selected_id = examples[random_id]['info']['choice']
                    answer_selected = examples[selected_id]['summaries'][0]['text']
                breakpoint()
        """

    # Dataset - Instruct GPTJ
    raw_datasets_gptj = load_dataset("Dahoas/synthetic-instruct-gptj-pairwise", cache_dir="./hf_cache/")
    raw_datasets_gptj = raw_datasets_gptj.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
    raw_datasets_gptj = raw_datasets_gptj['train']
    total_size = len(raw_datasets_gptj)
    train_size = int(0.9 * total_size)
    train_dataset_gptj = raw_datasets_gptj.select(range(train_size))
    eval_dataset_gptj = raw_datasets_gptj.select(range(train_size, total_size))

    # Dataset - WebGPT
    raw_datasets_webgpt = load_dataset("openai/webgpt_comparisons", cache_dir="./hf_cache/")
    raw_datasets_webgpt = raw_datasets_webgpt.map(
        preprocess_function_webgpt,
        batched=True,
        num_proc=4,
    )
    raw_datasets_webgpt = raw_datasets_webgpt['train']
    total_size = len(raw_datasets_webgpt)
    train_size = int(0.9 * total_size)
    train_dataset_webgpt = raw_datasets_webgpt.select(range(train_size))
    eval_dataset_webgpt = raw_datasets_webgpt.select(range(train_size, total_size))

    # Dataset - HH RLHF
    raw_datasets_rlhf = load_dataset("Anthropic/hh-rlhf", cache_dir="./hf_cache/")
    raw_datasets_rlhf = raw_datasets_rlhf.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
    train_dataset_rlhf = raw_datasets_rlhf["train"]
    eval_dataset_rlhf = raw_datasets_rlhf["test"]

    raw_datasets_summ = load_dataset("openai/summarize_from_feedback", "comparisons", cache_dir="./hf_cache/")
    paired_summ_datasets = {} 
    paired_summ_datasets_train = convert_summarize_to_pairwise(raw_datasets_summ['train'])
    paired_summ_datasets_train = Dataset.from_list(paired_summ_datasets_train)
    paired_summ_datasets_validation = convert_summarize_to_pairwise(raw_datasets_summ['validation'])[:20000]
    paired_summ_datasets_validation = Dataset.from_list(paired_summ_datasets_validation)
    paired_summ_datasets = DatasetDict({
            'train': paired_summ_datasets_train,
            'validation': paired_summ_datasets_validation,
            })
    paired_summ_datasets = paired_summ_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
    train_dataset_summ = paired_summ_datasets["train"]
    eval_dataset_summ = paired_summ_datasets["validation"]

    # Combine datasets
    train_dataset = concatenate_datasets([train_dataset_gptj, train_dataset_webgpt, train_dataset_rlhf]) #, train_dataset_summ])
    eval_dataset = concatenate_datasets([eval_dataset_gptj, eval_dataset_webgpt, eval_dataset_rlhf]) #, eval_dataset_summ])

    print("Pre-filtering train: ", len(train_dataset))
    print("Pre-filtering eval: ", len(eval_dataset))

    # for debugging
    # train_dataset = [train_dataset_gptj[0], train_dataset_webgpt[0], train_dataset_rlhf[0]]
    # eval_dataset = [eval_dataset_gptj[0], eval_dataset_webgpt[0], eval_dataset_rlhf[0]]

    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= reward_config.max_length
        and len(x["input_ids_rejected"]) <= reward_config.max_length
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= reward_config.max_length
        and len(x["input_ids_rejected"]) <= reward_config.max_length
    )
    print("Post-filtering train: ", len(train_dataset))
    print("Post-filtering eval: ", len(eval_dataset))

    ################
    # Training
    ################
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_config),
    )
    trainer.train()
    trainer.save_model(reward_config.output_dir)

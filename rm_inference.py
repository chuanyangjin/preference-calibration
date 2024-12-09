import random
random.seed(42)
import openai
from tqdm import tqdm
import json
import math
import os
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd

def generate_response(prompt, max_tokens = 10, temperature = 0):
    response = openai.chat.completions.create(
        model = "gpt-4",    # choose in "gpt-3.5-turbo" or "gpt-4"
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens = max_tokens,
        temperature = temperature,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    SHOW_ONLY = False
    datasets = ['webgpt', 'gptj', 'hh-rlhf', 'summarize']
    # datasets = ['ultrafeedback']
    num_examples_each_dataset = 25

    questions = []
    answers_0 = []
    answers_1 = []
    logs = []
    results = []
    ids = []

    READ_FROM_DATASET = False
    READ_FROM_CSV = True
    READ_OOD_DATA = True
    if READ_FROM_CSV:
        our_df = pd.read_csv('Feedback calibration - OUR_RATINGS.csv', skiprows=2)
        our_df = our_df.loc[:, ~our_df.columns.str.contains('^Unnamed')]    # drop unnamed columns
        print(our_df.shape)
        for i in range(0, 100):
            question = our_df.loc[i, 'question']
            answer_0 = our_df.loc[i, 'answer_0']
            answer_1 = our_df.loc[i, 'answer_1']
            this_id = our_df.loc[i, 'Input.ID']
            questions.append(question)
            answers_0.append(answer_0)
            answers_1.append(answer_1)
            ids.append(this_id)
        
        if READ_OOD_DATA:
            for i in range(100, 150):
                question = our_df.loc[i, 'question']
                answer_0 = our_df.loc[i, 'answer_0']
                answer_1 = our_df.loc[i, 'answer_1']
                this_id = our_df.loc[i, 'Input.ID']
                questions.append(question)
                answers_0.append(answer_0)
                answers_1.append(answer_1)
                ids.append(this_id)


    if not SHOW_ONLY:
        gpt_differences = []
        rm_differences = []
        rm_difference_probs = []
        reward_name = sys.argv[1]
        rank_model = AutoModelForSequenceClassification.from_pretrained(reward_name, cache_dir="./hf_cache/")
        tokenizer = AutoTokenizer.from_pretrained(reward_name, cache_dir="./hf_cache/")
        for question, answer_0, answer_1, question_id in zip(questions, answers_0, answers_1, ids):

            answer_0_tokens = tokenizer(question, answer_0, return_tensors='pt')
            # print(rank_model(**answer_0_tokens))
            rm_score_0 = rank_model(**answer_0_tokens).logits[0].cpu().detach().item()
            answer_1_tokens = tokenizer(question, answer_1, return_tensors='pt')
            rm_score_1 = rank_model(**answer_1_tokens).logits[0].cpu().detach().item()
            rm_difference = round(rm_score_0 - rm_score_1, 2)
            rm_differences.append(rm_difference)
            # print(f"Reward model score difference: {rm_difference}")

            # p_ij = (exp(r_i - r_j)/1+exp(r_i - r_j)))
            rm_difference_prob = math.exp(rm_score_0 - rm_score_1) / (1 + math.exp(rm_score_0 - rm_score_1))
            rm_difference_prob = round(rm_difference_prob, 2)
            rm_difference_probs.append(rm_difference_prob)
            # print(f"Reward model prob difference: {rm_difference_prob}")
            print(question_id)
            results.append({'question':question, 'answer_0':answer_0, 'answer_1':answer_1, 'reward_model_prob':rm_difference_prob, 'id':question_id})
            # idx+=1

        print(len(results))
        with open(sys.argv[2], "w") as f:
            for item in results:
                f.write(json.dumps(item)+'\n')


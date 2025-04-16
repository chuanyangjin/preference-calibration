import random
random.seed(42)
import openai
from tqdm import tqdm
import json
import math
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd

openai.api_key = os.environ["OPENAI_API_KEY"]

def generate_response(prompt, max_tokens = 10, temperature = 1.0):
    response = openai.chat.completions.create(
        model = "gpt-4-turbo-2024-04-09",    # choose in "gpt-3.5-turbo" or "gpt-4"
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens = max_tokens,
        temperature = temperature,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    datasets = ['webgpt', 'gptj', 'hh-rlhf', 'summarize', 'ultrafeedback']
    # datasets = ['ultrafeedback']

    questions = []
    answers_0 = []
    answers_1 = []
    logs = []
    results = []
    ids = []

    SHOW_ONLY = False
    READ_FROM_DATASET = False
    READ_FROM_CSV = True
    READ_OOD_DATA = True
    if READ_FROM_DATASET:
        for dataset in datasets:
            print(dataset)
            idx = 0
            if dataset == 'webgpt':
                data = load_dataset("openai/webgpt_comparisons", cache_dir="./hf_cache/")
            elif dataset == 'gptj':
                data = load_dataset("Dahoas/synthetic-instruct-gptj-pairwise", cache_dir="./hf_cache/")
            elif dataset == 'hh-rlhf':
                data = load_dataset("Deojoandco/anthropic-hh-rlhf", cache_dir="./hf_cache/")
            elif dataset == 'summarize':
                data = load_dataset("openai/summarize_from_feedback", "comparisons", cache_dir="./hf_cache/")
            elif dataset == 'ultrafeedback':
                # data = load_dataset("openbmb/UltraFeedback", cache_dir="./hf_cache/")
                data = load_dataset("HuggingFaceH4/ultrafeedback_binarized", cache_dir="./hf_cache/")

            i = -1
            while len(questions) < num_examples_each_dataset * (datasets.index(dataset) + 1):
                i += 1
                if dataset == 'webgpt':
                    question = data['train'][i]['question']['full_text']
                    answer_0 = data['train'][i]['answer_0']
                    score_0 = data['train'][i]['score_0']
                    answer_1 = data['train'][i]['answer_1']
                    score_1 = data['train'][i]['score_1']
                elif dataset == 'gptj':
                    question = data['train'][i]['prompt']
                    answer_0 = data['train'][i]['chosen']
                    answer_1 = data['train'][i]['rejected']
                elif dataset == 'hh-rlhf':
                    question = data['train'][i]['prompt']
                    answer_0 = data['train'][i]['chosen']
                    answer_1 = data['train'][i]['rejected']
                elif dataset == 'summarize':
                    id_start = i
                    while data['train'][i]['info']['id'] == data['train'][i+1]['info']['id']:
                        i += 1
                    id_end = i
                    if id_end - id_start >= 1:
                        id_0, id_1 = random.sample(range(id_start, id_end + 1), 2)
                        assert (data['train'][id_0]['info']['post'] == data['train'][id_1]['info']['post'])
                        question = data['train'][id_0]['info']['post'] + '\n\nPlease summarize the text above for me.'
                        answer_0 = data['train'][id_0]['summaries'][0]['text']
                        answer_1 = data['train'][id_1]['summaries'][0]['text']
                    else:
                        continue
                elif dataset == 'ultrafeedback':
                    question = data['train_prefs'][i]['prompt']
                    answer_0 = data['train_prefs'][i]['chosen'][1]['content']
                    # print(answer_0)
                    answer_1 = data['train_prefs'][i]['rejected'][1]['content']
                
                # Invalid answer
                if answer_0.strip("\n ") == '' or answer_1.strip("\n ") == '':
                    continue
                
                # Random order
                if random.random() < 0.5:
                    answer_0, answer_1 = answer_1, answer_0

                questions.append(question)
                answers_0.append(answer_0)
                answers_1.append(answer_1)
                ids.append(dataset+str(idx))
                idx+=1

                if SHOW_ONLY:
                    print(f"Question {len(questions)}, Dataset {dataset}")
                    print(question)
                    print('Answer 0:')
                    print(answer_0)
                    print('Answer 1:')
                    print(answer_1)
                    print()

    if READ_FROM_CSV:
        our_df = pd.read_csv('Feedback calibration - OUR_RATINGS.csv', skiprows=2)
        our_df = our_df.loc[:, ~our_df.columns.str.contains('^Unnamed')]    # drop unnamed columns
        print(our_df.shape)
        
        for i in range(150):
            question = our_df.loc[i, 'question']
            answer_0 = our_df.loc[i, 'answer_0']
            answer_1 = our_df.loc[i, 'answer_1']
            qid = our_df.loc[i, 'Input.ID']
            if type(qid) != str:
                print(question, answer_0, answer_1, qid)
                continue
            questions.append(question)
            answers_0.append(answer_0)
            answers_1.append(answer_1)
            ids.append(qid)

    if not SHOW_ONLY:

        gpt_differences = []
        rm_differences = []
        rm_difference_probs = []
        for question, answer_0, answer_1, question_id in zip(questions, answers_0, answers_1, ids):
            print(question_id)
            preference_prompt_prefix = f"I am going to give you a prompt and two answers. \
                The goal is to identify the better answer. If the prompt is a question, we want the factually correct answer. In a conversation, we want helpful replies that do not cause harm. \
                I only want you to reply with only a single number score 0 or 1 based on the preferred answer. If the preferred answer is Answer A, return score 0 and if the preferred answer is Answer B then return score 1. \n\n\
                "
            preference_prompt_suffix = f"Prompt: {question} \n\
                Answer A: {answer_0} \n\
                Answer B: {answer_1} \n\
                Score: "
            
            preference_prompt = preference_prompt_prefix + preference_prompt_suffix
            # print(preference_prompt)
            all_responses = []
            n_responses = 10
            for i in range(n_responses):
                try:
                    response = generate_response(preference_prompt, temperature = 1.0)
                    try:
                        score = int(response)
                    except:
                        if "score" in response.lower():
                            score = int(response.split()[-1])
                        else:
                            print("Couldn't cast score to float: ", response)
                            score = None
                    all_responses.append(score)
                    # print(response)
                    # print(score)
                    # breakpoint()
                    logs.append({'prompt': preference_prompt, 'response': score, 'str_response':response, 'category': 'preference', 'question':question, 'answer_0':answer_0, 'answer_1':answer_1, 'id':question_id})
                except Exception as e:
                    print(f"An error occurred: {e}")
                    continue
            score = sum(all_responses)
            gpt_difference = 2 * score - 10     # score - (10 - score)
            gpt_differences.append(gpt_difference)
            # print(all_responses, score)
            # breakpoint()
            # print(f"GPT score difference: {gpt_difference}")

            # try: 
            #     response = generate_response(subjectivity_prompt)
            #     logs.append({'prompt': subjectivity_prompt, 'response': response, 'category': 'subjectivity', 'question':question, 'answer_0':answer_0, 'answer_1':answer_1, 'id':question_id})
            # except Exception as e:
            #     print(f"An error occurred: {e}")
            #     continue
            # print(f"Subjectivity: {response}")
            # print()

            # print(gpt_difference, ";", rm_difference_prob, ";", response)
            results.append({'question':question, 'answer_0':answer_0, 'answer_1':answer_1, 'all_responses':all_responses, 'gpt_difference':gpt_difference, 'id':question_id})
            # idx+=1

        with open("all_logs_4_binary.jsonl", "w") as f:
            for log in logs:
                f.write(json.dumps(log)+'\n')

        with open("all_results_4_binary.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r)+'\n')

        # plt.scatter(gpt_differences, rm_differences, color='blue', marker='o')
        # plt.xlabel('GPT score difference')
        # plt.ylabel('Reward model score difference')
        # plt.legend()
        # plt.show()

        # plt.scatter(gpt_differences, rm_difference_probs, color='blue', marker='o')
        # plt.xlabel('GPT: difference in scores')
        # plt.ylabel('Reward model: difference in probabilities')
        # plt.legend()
        # plt.grid(True)
        # plt.savefig('ScoreCorrelations_2.pdf', format='pdf')
        #plt.show()

import sys
import os
sys.path.append('../')

from popper.benchmark import discovery_bench_hypothesis
from popper.agent import SequentialFalsificationTest
from sklearn.metrics import accuracy_score, average_precision_score

from tqdm import tqdm
import argparse
import traceback
import time
import json
import pickle
import numpy as np


argparser = argparse.ArgumentParser()
argparser.add_argument("--exp_name", type=str, default="discovery_bench")
argparser.add_argument("--model", type=str, default="claude-3-5-sonnet-20241022")
argparser.add_argument('--samples', type=int, default=50)
argparser.add_argument('--num_tests', type=int, default=3)
argparser.add_argument('--starts_from', type=int, default=0)
argparser.add_argument('--permute', action='store_true', default=False)
argparser.add_argument('--llm_approx', action='store_true', default=False)
argparser.add_argument('--e_value', action='store_true', default=False)
argparser.add_argument('--react', action='store_true', default=False)
argparser.add_argument('--relevance_checker', action='store_true', default=False)

args = argparser.parse_args()


exp_name = args.exp_name

exp_name += args.model

exp_name += f'_{args.num_tests}tests'

if args.permute:
    # data_loader.permute_selected_columns()
    exp_name+='_permuted'

if args.llm_approx:
    exp_name+='_llm_approx'

if args.e_value:
    exp_name+='_e_value'

if args.react:
    exp_name+="_react"

if args.relevance_checker:
    exp_name+="_relevance_checker"
    
print(f"Running {exp_name}")

res = []
samples = args.samples
bm = discovery_bench_hypothesis(num_samples = samples)
predictions = []
targets = []

start = time.time()

#response = []
for i, example in tqdm(enumerate(bm.get_iterator()), total=samples, desc="Processing"):
    if i < args.starts_from:
        print(f"Skipping example {i}")
        continue
    try:
        data_loader = example["data_loader"]
        
        permuted = "not permuted" if example["answer"] else "permuted"
        print("======================================================")
        print(f'Processing {(example["task"], example["metadataid"], example["query_id"], permuted)}')
        for name, df in data_loader.table_dict.items():
            print(name)
            print(df.head())
        print("------------------------------------")
        
        agent = SequentialFalsificationTest(llm = args.model)
        if args.llm_approx:
            agent.configure(data = data_loader, alpha = 0.1, beta = 0.1, aggregate_test = 'LLM_approx', max_num_of_tests = args.num_tests, max_retry = 5, time_limit = 2, llm_approx = True, domain=example['domain'], relevance_checker=args.relevance_checker)
        else:
            if args.e_value:
                agent.configure(data = data_loader, alpha = 0.1, beta = 0.1, aggregate_test = 'E-value', max_num_of_tests = args.num_tests, max_retry = 5, time_limit = 2, domain=example['domain'], relevance_checker=args.relevance_checker, use_react_agent=args.react, max_failed_tests=args.num_tests)
            else:
                agent.configure(data = data_loader, alpha = 0.1, beta = 0.1, aggregate_test = 'Fisher', max_num_of_tests = args.num_tests, max_retry = 5, time_limit = 2, domain=example['domain'], relevance_checker=args.relevance_checker, max_failed_tests=args.num_tests)

        log, last_message, parsed_result = agent.go(example['prompt'])
        predictions.append((agent.res_stat, agent.res))
        targets.append(example['answer'])
        res.append({
            "task": example["task"],
            "metadataid": example["metadataid"],
            "query_id": example["query_id"],
            "log": log,
            "last_message": last_message,
            "parsed_result": parsed_result,
            "res": agent.res,
            "res_stat": agent.res_stat,
            "answer": example["answer"]
        })
        # res[(example["task"], example["metadataid"], example["query_id"])] = (log, last_message, parsed_result, agent.res, agent.res_stat, example['answer'])
    except Exception as e:
        print(f"Error for prompt '{example['prompt']}': {e}")
        print(traceback.format_exc())  # Print the full traceback for debugging
        res.append({
            "task": example["task"],
            "metadataid": example["metadataid"],
            "query_id": example["query_id"],
            "error": traceback.format_exc()
        })
        # res[(example["task"], example["metadataid"], example["query_id"])] = ('Error', traceback.format_exc())
        # predictions.append((0.0, False))
        continue

output_path = os.path.join(os.getcwd(), 'data/' + exp_name + '.json')

# with open(os.path.join(os.getcwd(), 'data/' + exp_name + '.pkl'), 'wb') as f:
with open(output_path, 'w') as f:
    # pickle.dump(res, f)
    json.dump(res, f, indent=4)

print(f"Results saved to {output_path}")
print("------------------------------------")
print("------------------------------------")

end = time.time()
print(f"Total elapsed time: {end - start}")

print("Benchmark Results:")
print(f"Predictions: {predictions}")
print(f"Targets: {targets}")
eval_results = bm.evaluate(predictions, targets)

for metric in eval_results:
    print(f"{metric}: {eval_results[metric]}")
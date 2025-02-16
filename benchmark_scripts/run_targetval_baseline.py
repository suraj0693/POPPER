import sys
import os
sys.path.append('../')

from baseline_agents.coder_agent import BaseAgent
from baseline_agents.react_agent import ReactAgent
from baseline_agents.self_refine_agent import SelfRefineAgent
from baseline_agents.coder_utils import load_data_to_coder_globals
from baseline_agents.react_utils import load_data_to_react_globals
from popper.benchmark import gene_perturb_hypothesis
from popper.utils import ExperimentalDataLoader
from langchain_core.prompts import ChatPromptTemplate
from popper.utils import get_llm
from sklearn.metrics import accuracy_score, average_precision_score
from pydantic import BaseModel, Field
from typing import (
    Optional, List, Tuple, Union, Literal, Dict, TypedDict, Annotated
)
from tqdm import tqdm
import argparse
import traceback
import time
import json

argparser = argparse.ArgumentParser()
argparser.add_argument("--exp_name", type=str, default="gene_baseline")
argparser.add_argument("--model_name", type=str, default="claude-3-5-sonnet")
argparser.add_argument("--agent_type", type=str, choices=['coder', 'react', 'self_refine'], default="coder")
argparser.add_argument('--samples', type=int, default=25)
argparser.add_argument('--starts_from', type=int, default=0)
argparser.add_argument("--log_file", type=str, default=".logs/baseline_log.log")
argparser.add_argument('--permute', action='store_true', default=False)
argparser.add_argument('--use_full_data', action='store_true', default=False)
argparser.add_argument("--dataset", type=str, default="IL2")
argparser.add_argument('--seed', type=int, default=-1)
argparser.add_argument('--use_simple_template', action='store_true', default=False)
argparser.add_argument('--path', type=str, required = True)

args = argparser.parse_args()

data_path = args.path
if args.use_full_data:
    data_loader = ExperimentalDataLoader(data_path, table_dict_selection = 'all_bio')
else:
    data_loader = ExperimentalDataLoader(data_path, table_dict_selection = 'default')

exp_name = args.dataset
exp_name += f"_{args.agent_type}"
if args.permute:
    data_loader.permute_selected_columns()
    exp_name+='_permuted'

if args.use_full_data:
    exp_name+='_full_data'

if args.seed != -1:
    exp_name+=f'_seed_{args.seed}'

if args.use_simple_template:
    exp_name+='_simple_template'

samples = args.samples
bm = gene_perturb_hypothesis(num_of_samples = samples,
permuted=args.permute, dataset = args.dataset, path = args.path)
predictions = []
targets = []

agent = None
load_to_globals = None
if args.agent_type == 'coder' or args.agent_type == 'react':
    load_to_globals = load_data_to_coder_globals if args.agent_type == 'coder' else load_data_to_react_globals

start = time.time()
for i, example in tqdm(enumerate(bm.get_iterator()), total=samples, desc="Processing"):
    if i < args.starts_from:
        print(f"Skipping example {i}")
        continue
    
    if args.agent_type == 'coder':
        agent = BaseAgent(
            model_name=args.model_name,
            api_config="baseline_agents/config/api_config.json",
            model_config="baseline_agents/config/model_config.json",
            log_file=args.log_file,
            simple_prompt=args.use_simple_template
        )
    elif args.agent_type == 'react':
        agent = ReactAgent(
            model_name=args.model_name,
            api_config="baseline_agents/config/api_config.json",
            model_config="baseline_agents/config/model_config.json",
            log_file=args.log_file,
            simple_prompt=args.use_simple_template
        )
    elif args.agent_type == 'self_refine':
        agent = SelfRefineAgent(
            llm="claude-3-5-sonnet-20241022"
        )
    else:
        raise ValueError(f"Agent {args.agent_type} not found")
    
    try:
        if load_to_globals is not None:
            load_to_globals(data_loader)
        
        query = example['prompt']
        datasets = data_loader.data_desc
        if args.agent_type == 'coder':
            output = agent.generate(data_loader=data_loader, query=query)
        elif args.agent_type in ['react', 'self_refine']:
            output = agent.generate(data_loader=data_loader, query=query)
    
        print(output)
        if output is not None:
            predictions.append((0.0, output["output"], example['gene']))
        else:
            predictions.append((0.0, False, example['gene']))
        targets.append(example['binary_answer'])
    except:
        print(traceback.format_exc())


print("------------------------------------")

end = time.time()
print(f"Total elapsed time: {end - start}")

print("Benchmark Results:")
print(f"Predictions: {predictions}")
print(f"Targets: {targets}")
eval_results = bm.evaluate(predictions, targets)

for metric in eval_results:
    print(f"{metric}: {eval_results[metric]}")

    
import pickle
import os
os.makedirs(args.path + '/res', exist_ok=True)
with open(args.path + '/res/' + exp_name + '_res_final.pkl', 'wb') as f:
    pickle.dump(res, f)
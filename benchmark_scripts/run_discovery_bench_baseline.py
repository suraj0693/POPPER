import sys
import os
sys.path.append(os.getcwd())

from baseline_agents.coder_agent import BaseAgent
from baseline_agents.react_agent import ReactAgent
from baseline_agents.self_refine_agent import SelfRefineAgent
from baseline_agents.coder_utils import load_data_to_coder_globals
from baseline_agents.react_utils import load_data_to_react_globals
from popper.benchmark import discovery_bench_hypothesis

from sklearn.metrics import accuracy_score, average_precision_score

from tqdm import tqdm
import argparse
import traceback
import time
import json

argparser = argparse.ArgumentParser()
argparser.add_argument("--exp_name", type=str, default="discovery_bench_baseline")
argparser.add_argument("--model_name", type=str, default="claude-3-5-sonnet")
argparser.add_argument("--agent_type", type=str, choices=['coder', 'react', 'self_refine'], default="coder")
argparser.add_argument('--samples', type=int, default=50)
argparser.add_argument('--starts_from', type=int, default=0)
argparser.add_argument("--log_file", type=str, default=".logs/baseline_log.log")
argparser.add_argument('--permute', action='store_true', default=False)

args = argparser.parse_args()

exp_name = args.exp_name
exp_name += f"_{args.agent_type}"
if args.permute:
    # data_loader.permute_selected_columns()
    exp_name+='_permuted'

samples = args.samples
bm = discovery_bench_hypothesis(num_samples = samples)
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
            log_file=args.log_file
        )
    elif args.agent_type == 'react':
        agent = ReactAgent(
            model_name=args.model_name,
            api_config="baseline_agents/config/api_config.json",
            model_config="baseline_agents/config/model_config.json",
            log_file=args.log_file
        )
    elif args.agent_type == 'self_refine':
        agent = SelfRefineAgent(
            llm="claude-3-5-sonnet-20241022"
        )
    else:
        raise ValueError(f"Agent {args.agent_type} not found")
    
    try:
        data_loader = example["data_loader"]
        if load_to_globals is not None:
            load_to_globals(data_loader)
        
        query = example['prompt']
        output = agent.generate(data_loader=data_loader, query=query)
        if output is not None:
            predictions.append((0.0, output["output"]))
        else:
            predictions.append((0.0, False))
        targets.append(example['answer'])
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
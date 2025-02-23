import sys
import os
sys.path.append('../')

from popper.benchmark import gene_perturb_hypothesis
from popper.agent import SequentialFalsificationTest
from popper.utils import ExperimentalDataLoader

from tqdm import tqdm
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--permute', action='store_true', default=False)
argparser.add_argument('--samples', type=int, default=25)
argparser.add_argument('--llm_approx', action='store_true', default=False)
argparser.add_argument('--e_value', action='store_true', default=False)
argparser.add_argument('--relevance_checker', action='store_true', default=False)
argparser.add_argument('--react', action='store_true', default=False)
argparser.add_argument('--use_full_data', action='store_true', default=False)
argparser.add_argument('--max_num_of_tests', type=int, default=5)
argparser.add_argument('--alpha', type=float, default=0.1)
argparser.add_argument('--seed', type=int, default=-1)
argparser.add_argument("--model", type=str, default="claude-3-5-sonnet-20241022")
argparser.add_argument("--dataset", type=str, default="IL2")
argparser.add_argument('--data_sampling', type=int, default=-1)
argparser.add_argument('--user_study_neg_genes', action='store_true', default=False)
argparser.add_argument('--is_locally_served', action='store_true', default=False)
argparser.add_argument('--server_port', type=int, required=False)
argparser.add_argument("--api_key", type=str, default="EMPTY")
argparser.add_argument('--path', type=str, required = True)

args = argparser.parse_args()

data_path = args.path

if args.data_sampling != -1:
    data_loader = ExperimentalDataLoader(data_path, table_dict_selection = 'all_bio', data_sampling = args.data_sampling)
elif args.use_full_data:
    data_loader = ExperimentalDataLoader(data_path, table_dict_selection = 'all_bio')
else:
    data_loader = ExperimentalDataLoader(data_path, table_dict_selection = 'default')

exp_name = args.dataset
if args.permute:
    data_loader.permute_selected_columns()
    exp_name+='_permuted'

if args.max_num_of_tests != 5:
    exp_name+=f'_max_{args.max_num_of_tests}'

if args.alpha != 0.1:
    exp_name+=f'_alpha_{args.alpha}'

if args.llm_approx:
    exp_name+='_llm_approx'

if args.use_full_data:
    exp_name+='_full_data'

if args.react:
    exp_name+="_react_v2"
    #args.e_value = True

if args.e_value:
    exp_name+='_e_value'

if args.relevance_checker:
    exp_name+='_relevance_checker'

if args.seed != -1:
    exp_name+=f'_seed_{args.seed}'

if args.model[:6] != 'claude':
    exp_name+=f'_{args.model}'
else:
    if 'haiku' in args.model:
        exp_name+='_haiku'
        
if args.data_sampling != -1:
    exp_name+=f'_sampling_{args.data_sampling}'

if args.user_study_neg_genes:
    exp_name+='_user_study_neg_genes'

print('Experiment name: ' + exp_name)
res = {}
samples = args.samples
bm = gene_perturb_hypothesis(num_of_samples = samples, permuted=args.permute, 
dataset = args.dataset, user_study_neg_genes= args.user_study_neg_genes, path = args.path)
#response = []
for example in tqdm(bm.get_iterator(), total=samples, desc="Processing"):
    import traceback
    try:
        agent = SequentialFalsificationTest(llm = args.model, is_local=args.is_locally_served, port=args.server_port, api_key=args.api_key)
        if args.llm_approx:
            agent.configure(data = data_loader, 
                        alpha = args.alpha, beta = 0.1, 
                        aggregate_test = 'LLM_approx', 
                        max_num_of_tests = 5, 
                        max_retry = args.max_num_of_tests, time_limit = 2, 
                        llm_approx = True, 
                        relevance_checker = args.relevance_checker)
        else:
            if args.e_value:
                agent.configure(data = data_loader, alpha = args.alpha, 
                            beta = 0.1, aggregate_test = 'E-value', 
                            max_num_of_tests = args.max_num_of_tests, max_retry = 5, time_limit = 2, 
                            relevance_checker = args.relevance_checker, use_react_agent=args.react)
            else:
                agent.configure(data = data_loader, alpha = args.alpha, beta = 0.1, 
                                aggregate_test = 'Fisher', max_num_of_tests = args.max_num_of_tests,
                                max_retry = args.max_num_of_tests, time_limit = 2, 
                                relevance_checker = args.relevance_checker, use_react_agent=args.react)

        log, last_message, parsed_result = agent.go(example['prompt'])
        res[example['gene']] = (log, last_message, parsed_result, agent.res_stat)
    except Exception as e:
        print(f"Error for prompt '{example['prompt']}': {e}")
        print(traceback.format_exc())  # Print the full traceback for debugging
        res[example['gene']] = ('Error', traceback.format_exc())
        continue
    
import pickle
import os
os.makedirs(args.path + '/res', exist_ok=True)
with open(args.path + '/res/' + exp_name + '_res_final.pkl', 'wb') as f:
    pickle.dump(res, f)
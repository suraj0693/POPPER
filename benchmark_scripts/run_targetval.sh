## TargetVal benchmarks
data_path="/dfs/user/kexinh/popper_data_processed"
for seed in 1 2 3 4 5
do
datasets=("IFNG" "IL2")
for dataset in "${datasets[@]}"
do
    python run_targetval_benchmark.py --e_value --relevance_checker --react --seed $seed --use_full_data --samples 20 --dataset $dataset --path $data_path
    python run_targetval_benchmark.py --e_value --relevance_checker --permute --react --seed $seed --use_full_data --samples 50 --dataset $dataset --path $data_path
done
done

### For Popper-codegen: 
python run_targetval_benchmark.py --e_value --relevance_checker --seed $seed --use_full_data --samples 20 --dataset $dataset --path $data_path
python run_targetval_benchmark.py --e_value --relevance_checker --permute --seed $seed --use_full_data --samples 50 --dataset $dataset --path $data_path


### For Popper-NoRelevanceChecker:
python run_targetval_benchmark.py --e_value --react --seed $seed --use_full_data --samples 20 --dataset $dataset --path $data_path
python run_targetval_benchmark.py --e_value --permute --react --seed $seed --use_full_data --samples 50 --dataset $dataset --path $data_path

### For Popper-Fisher Combined Test:
python run_targetval_benchmark.py --relevance_checker --react --seed $seed --use_full_data --samples 20 --dataset $dataset --path $data_path
python run_targetval_benchmark.py --relevance_checker --permute --react --seed $seed --use_full_data --samples 50 --dataset $dataset --path $data_path

### For Popper-LLM estiamted likelihood ratio:
python run_targetval_benchmark.py --llm_approx --relevance_checker --seed $seed --use_full_data --samples 20 --dataset $dataset --path $data_path
python run_targetval_benchmark.py --llm_approx --relevance_checker --permute --seed $seed --use_full_data --samples 50 --dataset $dataset --path $data_path

### For other LLMs:
# GPT-4o
python run_targetval_benchmark.py --e_value --relevance_checker --react --seed $seed --use_full_data --model gpt-4o --samples 20 --dataset $dataset --path $data_path
python run_targetval_benchmark.py --e_value --relevance_checker --permute --react --seed $seed --use_full_data --model gpt-4o --samples 50 --dataset $dataset --path $data_path

# o1
python run_targetval_benchmark.py --e_value --relevance_checker --react --seed $seed --use_full_data --model o1-2024-12-17 --samples 20 --dataset $dataset --path $data_path
python run_targetval_benchmark.py --e_value --relevance_checker --permute --react --seed $seed --use_full_data --model o1-2024-12-17 --samples 50 --dataset $dataset --path $data_path

# Haiku
python run_targetval_benchmark.py --e_value --relevance_checker --react --seed $seed --use_full_data --model claude-3-5-haiku-20241022 --samples 20 --dataset $dataset --path $data_path
python run_targetval_benchmark.py --e_value --relevance_checker --permute --react --seed $seed --use_full_data --model claude-3-5-haiku-20241022 --samples 50 --dataset $dataset --path $data_path

### For user study type I error genes:
python run_targetval_benchmark.py --e_value --relevance_checker --permute --react --seed $seed --use_full_data --user_study_neg_genes --dataset IL2 --path $data_path


### Baselines
# Coder-o1
python run_targetval_baseline.py --agent_type coder --model o1-2024-12-17 --permute --samples 50 --seed $seed --dataset IL2 --path $data_path
python run_targetval_baseline.py --agent_type coder --model o1-2024-12-17 --samples 20 --seed $seed --dataset IL2 --path $data_path

# Coder
python run_targetval_baseline.py --agent_type coder --permute --samples 50 --seed $seed --dataset IL2 --path $data_path
python run_targetval_baseline.py --agent_type coder --samples 20 --seed $seed --dataset IL2 --path $data_path

# Self-refine
python run_targetval_baseline.py --agent_type self_refine --use_full_data --permute --samples 50 --seed $seed --dataset IL2 --path $data_path
python run_targetval_baseline.py --agent_type self_refine --use_full_data --samples 20 --seed $seed --dataset IL2 --path $data_path

# React
python run_targetval_baseline.py --use_other_claude_api --use_simple_template --agent_type react --permute --samples 50 --seed $seed --dataset IL2 --path $data_path
python run_targetval_baseline.py --use_other_claude_api --use_simple_template --agent_type react --samples 20 --seed $seed --dataset IL2 --path $data_path
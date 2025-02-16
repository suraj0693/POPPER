python benchmark_scripts/run_discovery_bench.py \
  --exp_name discovery_bench --num_tests 5 --samples 300 --permute --e_value --react --relevance_checker

python benchmark_scripts/run_discovery_bench_baseline.py \
  --exp_name discovery_bench_baseline_react --agent_type react --samples 50 --log_file .logs/discovery_bench_baseline_react.log --permute &> .logs/discovery_bench_baseline_react_stdout.log


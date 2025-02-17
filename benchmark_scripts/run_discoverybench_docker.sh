# docker build --no-cache -t discovery_benchmark .
docker build -t discovery_benchmark .


docker run \
  -v /dfs/scratch0/lansong/discoverybench:/dfs/scratch0/lansong/discoverybench \
  -v /dfs/scratch0/lansong/data:/app/data \
  --name discovery_bench \
  --env-file .env \
  discovery_benchmark benchmark_scripts/run_discovery_bench.py \
  --exp_name discovery_bench_v3 --num_tests 5 --samples 100 --permute --e_value --react --relevance_checker \
  --path /dfs/scratch0/lansong/discoverybench &> .logs/discovery_bench_evalue_react_v3.log


docker wait discovery_bench

docker rm discovery_bench

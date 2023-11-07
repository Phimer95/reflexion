python main.py \
  --run_name "reflexion" \
  --root_dir "root" \
  --dataset_path ./benchmarks/failing_subset.jsonl \
  --strategy "reflexion" \
  --language "py" \
  --model "gpt-3.5-turbo" \
  --pass_at_k "1" \
  --max_iters "4" \
  --verbose

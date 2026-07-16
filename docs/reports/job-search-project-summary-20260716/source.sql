-- Reviewed source rows for the job-search project summary.
-- Values were transcribed from PROJECT_LOG.md, docs/project-log/2026-07-13.md,
-- docs/project-log/2026-07-14.md, docs/project-log/2026-07-15.md,
-- config/serving.env, benchmarks/modelcard/README.md, and
-- benchmarks/modelcard/tau2_runtime_patch.py on 2026-07-16.
-- This DuckDB-compatible SQL reproduces the compact evidence datasets rendered
-- by artifact.json; narrative judgments and caveats remain in the report blocks.

-- Headline deployment, automation, and throughput evidence.
select
  4::integer as gpus,
  2::integer as models,
  20::integer as p1_smoke,
  14::integer as core_statuses,
  2342.2::double as qwen_tps,
  2354.3::double as thinkingcap_tps;

-- Role-fit visualization. Scores use an explicit 1–5 ordinal judgment based on
-- the documented project evidence and are not labor-market outcome statistics.
select *
from (values
  ('LLM 推理/部署', 5, '强', 'H100、vLLM、TP2、FP8、MTP、性能实验'),
  ('模型评测/Benchmark', 5, '强', '评测自动化、协议冻结、结果审计'),
  ('ML Systems / AI Infra', 4, '较强', '容器、GPU 隔离、离线依赖、任务恢复'),
  ('Agent / Tool-use', 4, '较强', 'BFCL、TAU2、tool parser、轨迹调试'),
  ('通用机器学习工程', 3, '中等', '实验设计、数据与环境治理'),
  ('模型训练/算法研究', 2, '偏弱', '只有推理与评测侧证据')
) as t(role, score, fit, top_evidence);

-- Quantitative result checks used in the report narrative.
select *
from (values
  ('steady_concurrency_16_qwen_tps', 2342.2, 'output tok/s'),
  ('steady_concurrency_16_thinkingcap_tps', 2354.3, 'output tok/s'),
  ('p1_real_smoke_successes', 20.0, 'model-task statuses'),
  ('core_full_successes', 14.0, 'model-task statuses'),
  ('ifeval_full_delta', 2.96, 'percentage points'),
  ('ifeval_nontruncated_delta', 1.20, 'percentage points'),
  ('longbench_completed', 400.0, 'samples'),
  ('longbench_total', 503.0, 'samples'),
  ('tau2_priority_total', 269.0, 'samples')
) as t(metric, value, unit);

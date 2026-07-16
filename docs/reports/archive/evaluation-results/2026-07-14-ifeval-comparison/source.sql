-- IFEval 双模型报告的可执行 SQLite 转换层。
-- 数值来自同目录 data.json；该 JSON 保存原始
-- EvalScope aggregate report、prediction JSONL 和 review JSONL 的相对路径。

CREATE TEMP TABLE metric_results (
  metric_order INTEGER,
  metric TEXT,
  qwen REAL,
  thinkingcap REAL,
  delta REAL,
  ci95_low REAL,
  ci95_high REAL,
  prompt_count INTEGER
);

INSERT INTO metric_results VALUES
  (1, 'Prompt strict',      0.8484288355, 0.8780036969, 0.0295748614, 0.0073937153, 0.0517560074, 541),
  (2, 'Instruction strict', 0.8801601972, 0.9063462723, 0.0261860752, 0.0064695009, 0.0468268638, 541),
  (3, 'Prompt loose',       0.8743068392, 0.9020332717, 0.0277264325, 0.0055452865, 0.0517560074, 541),
  (4, 'Instruction loose',  0.8998767714, 0.9251386322, 0.0252618608, 0.0052372150, 0.0459026494, 541);

CREATE TEMP TABLE runtime_results (
  model TEXT,
  completed INTEGER,
  errors INTEGER,
  mean_latency_s REAL,
  p90_latency_s REAL,
  mean_output_tokens REAL,
  max_token_stops INTEGER,
  wall_time_s INTEGER
);

INSERT INTO runtime_results VALUES
  ('Qwen3.6-27B-FP8', 541, 0, 11.581443, 20.788215, 2876.911275, 30, 1640),
  ('ThinkingCap-Qwen3.6-27B-FP8', 541, 0, 8.592124, 14.981783, 2010.162662, 21, 1241);

CREATE TEMP TABLE paired_prompt_results (
  mode TEXT,
  both_pass INTEGER,
  qwen_only INTEGER,
  thinkingcap_only INTEGER,
  both_fail INTEGER,
  mcnemar_exact_p REAL
);

INSERT INTO paired_prompt_results VALUES
  ('strict', 447, 12, 28, 54, 0.0165890034),
  ('loose', 460, 13, 28, 40, 0.0275331558);

CREATE TEMP TABLE truncation_sensitivity (
  cohort TEXT,
  prompt_count INTEGER,
  qwen_prompt_strict REAL,
  thinkingcap_prompt_strict REAL,
  delta REAL,
  ci95_low REAL,
  ci95_high REAL,
  qwen_only INTEGER,
  thinkingcap_only INTEGER,
  mcnemar_exact_p REAL
);

INSERT INTO truncation_sensitivity VALUES
  ('Full benchmark', 541, 0.8484288355, 0.8780036969, 0.0295748614, 0.0073937153, 0.0517560074, 12, 28, 0.0165890034),
  ('Exclude either-model max-token stops', 502, 0.8984063745, 0.9103585657, 0.0119521912, -0.0039840637, 0.0278884462, 6, 12, 0.2378845215);

CREATE TEMP TABLE instruction_type_results (
  instruction TEXT,
  n INTEGER,
  qwen_strict REAL,
  thinkingcap_strict REAL,
  delta REAL
);

INSERT INTO instruction_type_results VALUES
  ('change_case:english_capital', 25, 0.68, 0.80, 0.12),
  ('change_case:capital_word_frequency', 25, 0.92, 0.80, -0.12),
  ('detectable_format:number_highlighted_sections', 48, 0.8958333333, 0.9791666667, 0.0833333333),
  ('detectable_content:postscript', 26, 0.9230769231, 1.0, 0.0769230769),
  ('combination:repeat_prompt', 41, 0.7317073171, 0.8048780488, 0.0731707317),
  ('detectable_format:json_format', 17, 0.9411764706, 0.8823529412, -0.0588235294);

SELECT * FROM metric_results ORDER BY metric_order;
SELECT * FROM runtime_results ORDER BY model;
SELECT * FROM paired_prompt_results ORDER BY mode;
SELECT * FROM truncation_sensitivity ORDER BY prompt_count DESC;
SELECT *, ABS(delta) AS abs_delta
FROM instruction_type_results
ORDER BY abs_delta DESC, instruction;

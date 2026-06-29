---
title: "Conductor-Train pilot — データ仕様 / verifier 要件（VSCode handoff）"
type: "note"
status: "in-progress"
tags: ["learning", "llm", "reinforcement-learning", "dataset"]
created: 2026-06-29
updated: 2026-06-29
---

> [!INFO] このノートの位置づけ
> pilot RL データ（201件）の **schema・verifier 採点規則・gold 信頼性・OOD 根拠** を、VSCode の実装 Agent に渡せる形で固めた handoff。データ本体は `_assets/conductor-train-pilot/pilot.jsonl`。設計の親ノートは [[conductor-rl-自前構築メモ]]、eval/出典の grounding は [[fugu-eval-benchmarks-2026]] / [[conductor-train-task-sources-2026]]。

## 所在と構成

- `_assets/conductor-train-pilot/pilot.jsonl` — **201 件**（merge 済・bad JSON 0）
- per-cluster：`code.jsonl`(71) / `science_mcq.jsonl`(65) / `hard_math.jsonl`(65) ＋ 各 `*.manifest.json`
- 難度：easy 59 / med 80 / hard 62（**group の std>0 を保つため意図的に混在**＝Problem① 対策）
- **全件 `gold_confidence=1.0`**（生成時に検証済、下記）
- `source=generated` ＝ **OOD by construction**（新規生成なので Fugu eval に物理的に非含有。contamination ゼロ・gold 正しさだけが論点）

## item schema（JSONL・1 行 1 件）

| field | 型 | 説明 |
|---|---|---|
| `id` | str | `code-0001` / `sci-0001` / `math-0001` |
| `cluster` | str | `code` \| `science_mcq` \| `hard_math` |
| `prompt` | str | solver が見るタスク文（MCQ は本文＋A–D を内包） |
| `gold` | str/null | code は `null`（gold は tests が保持）／mcq は letter／math は sympy-parseable な最終解 |
| `verifier` | str | `code_exec` \| `mcq_exact` \| `math_verify` |
| `verifier_spec` | obj | cluster 別（下記） |
| `difficulty` | str | `easy` \| `med` \| `hard` |
| `source` | str | `generated` |
| `gen_method` | str | 生成法＋gold 検証法 |
| `gold_confidence` | float | 全件 1.0 |
| `eval_disjoint_note` | str | OOD 根拠 |

## verifier 採点規則（= `s_correct` の作り方・実装の核）

> reward の主項は `w_corr · s_correct`（[[conductor-rl-自前構築メモ]] 報酬設計）。`s_correct` は以下で算出。`f_fmt`（DAG 妥当）/ `f_exec`（呼び出し実行可能性）/ `b_eff`（速さ・安さ／正解時のみ）は **data ではなく executor 側**で計算。

- **`code_exec`**：候補コードを **secure sandbox**（ネット遮断・resource/time limit、`time_limit_s=5`）で実行。`verifier_spec.tests[].input` を stdin に与え stdout を `output` と比較。**正規化＝行ごと `rstrip` ＋ 全体 `rstrip`**（worker 出力にも同じ正規化を適用）。`s_correct` = 全 test pass の二値、**または通過テスト率の連続値**（hard cluster の信号消失を緩めるなら連続を推奨）。
- **`mcq_exact`**：候補から最終 letter を抽出（`answer is ([A-D])` / boxed 等）→ `gold` と一致で `s_correct=1`。偽陰性はほぼ**抽出失敗**由来なので抽出を頑健に。
- **`math_verify`**：候補から `\boxed{}` 最終解を抽出 → **SymPy 等価判定**（`verifier_spec.tolerance=1e-6`）。等価形の偽陰性（`1/2` vs `0.5`、未簡約）対策に **TinyV 風の軽量 LLM-verifier を後段 fallback**。

## gold の信頼性（どう検証したか・残存リスク）

- **code**：参照解を実行して期待出力を構成（gold は構成的）＋ **weak-test チェック**（自明な誤答が ≥1 test で落ちることを確認）＋ 全 **566 test を再実行監査して 0 不一致**。⚠ 残存：参照解が微妙に誤ると self-consistent な誤 gold。規約が効く 2 件（`code-0035` 負数の floor/truncation、`code-0070` 到着==出発の同時不可）は prompt に明記済。全件 `stdin_stdout`（function 形式なし）。
- **science_mcq**：意図 key を見ずに**独立再解答**し照合（定量は SymPy 計算）。⚠ 残存：概念問題が数件あり first-principles 照合（暗記でも解け得る）。純粋な名称想起 recall は最初から除外。初稿で正解が B に偏る hack を検出し **id seed の決定的シャッフルで是正**（A14/B15/C18/D18）。
- **hard_math**：**全 65 件 SymPy 計算**（self-consistency 不使用）＝最も信頼できる。⚠ 残存：競技技法ゆえ答が既知定数と偶然一致しうる（特定問題の漏洩ではない）。

## 既知の運用注意

- **出力正規化を grader と worker で一致**させる（line `rstrip` ＋ overall `rstrip`）。不一致は偽陰性の温床。
- code の tests は**小入力**＝TLE/perf 回帰は検出しない（pilot 用 reward には十分）。
- 難度ラベルはヒューリスティック（計算/概念の深さ）。

## スケール時（generated → curate）

pilot は生成（OOD by construction・gold 検証済だが **reference-defined**）。本訓練でスケールする際は研究の推奨ソースへ（[[conductor-train-task-sources-2026]]）：

- code → **CodeContests**（+ HARDTESTS で頑健テスト）・**temporal split**（旧 window 訓練／最新 LiveCodeBench を eval）
- science → **SuperGPQA**、hard → **OlympiadBench / Omni-MATH**（SymPy 報酬を再利用）
- charts(MM・pilot-2) → **ReachQA** ＋ synthetic-matplotlib
- **本 pilot の verifier 群はそのまま再利用可能**（schema 互換）。

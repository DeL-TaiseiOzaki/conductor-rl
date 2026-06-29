
> [!INFO] このノートについて
> Claude.ai での議論を Learning に蒸留したもの。**Sakana Fugu / Trinity / Conductor の理解**から**自前 Conductor（Qwen3.5-4B + OpenRouter ワーカー）の構築設計**までを確定させた回。議論はここで「次は実装」という地点で止まっている＝**ここから再開する**。
> 生ログ全文：[[arxiv論文の解説リクエスト-aec729b9]]（`Inbox/2026-06-29/chat-logs/`）

---

## 🎯 ゴール（やりたいこと）

`Qwen/Qwen3.5-4B` を **Conductor（指揮者）役**にして、OpenRouter 上の複数ワーカーモデルを**自由に組み合わせ**（単純に会話 / あるモデルに Web 検索させて後続でまとめる / 複数モデルの推論を1モデルが集約 …）させ、**タスク性能が最大になる組み合わせを Agent-use RL で学習**させる。制約は「**早く・正確に**解く」こと。

土台は Sakana AI の **Conductor / Trinity**（後述）の設計思想で、「桁違いに小さなオーケストレーターが巨大ワーカー群を指揮して集合知を引き出す」路線をなぞる。

---

## 🎯 ターゲット：Fugu-Ultra を eval で上回る（2026-06-29 確定）

最終目標は **Fugu-Ultra を同じ評価スイートで上回ること**。Fugu の eval ベンチ＝**SWE-Bench Pro / Terminal Bench 2.1 / LiveCodeBench / GPQA-Diamond / Humanity's Last Exam / CharXiv**。

**設計原則：train-to-eval competency alignment.** eval ベンチで直接訓練しない（contamination＋コスト）。各 eval が要求する**オーケストレーション能力**を，**auto-gradable な代理タスク**で誘導する。Conductor が学ぶのは「問題を解く」ことではなく「正解に至る指揮（DAG）を書く」ことなので，代理タスクが正しい指揮スキルを誘発すれば eval に転移する。

| Fugu eval | 要求される能力 | 訓練代理（auto-gradable・grounding 中） | 誘導される指揮スキル |
|---|---|---|---|
| LiveCodeBench | 競技コーディング | 旧版 LiveCodeBench / CodeContests（time-split で contamination 回避） | builder↔debugger 分業，code-strong worker(V4 Pro)へ |
| GPQA-Diamond | 難 science QA | GPQA train / MMLU-Pro(STEM) / SciBench | science を GLM/V4 Pro へ，verify 役 |
| HLE | 超難・多領域 | Olympiad/AGIEval 系の closed-form 難問 | 最難ルーティング＋多 worker 集約 |
| CharXiv（**MM**） | 図表理解 | 検証可能 chart-QA（ChartQA/PlotQA/合成 matplotlib） | **図読みを M3（唯一の MM）へ** |
| SWE-Bench Pro（**agentic**） | repo SWE | SWE-Gym 等の実行検証パッチ（要 sandbox） | 多ターン builder/debugger，tool use |
| Terminal Bench 2.1（**agentic**） | 端末 agentic | サンドボックス shell タスク（pass/fail oracle） | 逐次計画，tool use を M3 へ |

**フェーズ（推奨）**：Phase 1＝静的検証＋マルチモーダル（LiveCodeBench/GPQA/HLE/chart-QA）で GRPO を立ち上げ → Phase 2＝agentic 実行環境（SWE/Terminal）を追加。**GRPO 直行（方式）と benchmark 網羅（範囲）は別軸**で，範囲だけ段階化する。

> [!WARNING] 戦略上の正直な注意
> Fugu のワーカーは**フロンティア（Gemini/GPT/Opus）**。本構成は**安価な OSS ワーカー（DeepSeek/MiniMax/GLM）**に差し替えている。同じ eval で**絶対スコア**で勝つには，オーケストレーションの利得が**ワーカー品質の差**を超える必要があり相当ハード。現実的な勝ち筋は ①**コスト効率フロンティア**（同等品質を桁違いに安く），②OSS が既にフロンティア近接な領域（**コーディング**：V4 Pro 等）での絶対勝ち。HLE 等の最難領域で絶対勝ちを狙うなら，最難サブタスク用にフロンティアワーカーを1枚だけ許す設計余地を残す。

> [!INFO] grounding 研究（完了・2026-06-29）
> Fugu eval 6本の精密 spec ＝ [[fugu-eval-benchmarks-2026]]（`.claude/docs/research/`）／ 訓練データ出典 ＝ [[conductor-train-task-sources-2026]]。下の「Conductor-Train」節がその grounded 版。**2026 era の arxiv/HF は preprint 級**として扱い、GRPO 設定を固める前に公式 repo で再確認する。

---

## 🧪 Conductor-Train：訓練タスクスイート設計（grounded・2026-06-29）

### 戦略：どこで Fugu-Ultra に勝てるか（headroom 分析）

- **GPQA-Diamond は飽和**（Fugu 95.5・N=198 → 約9問=ノイズ）。**差別化にならない**ので網羅目的で入れるが投資しない。
- **HLE（~50%）と SWE-Bench Pro（~74%）に実 headroom**。ここが本当に勝ちにいく場所。
- OSS ワーカーで**絶対勝ち**が現実的なのは **coding**（V4 Pro がフロンティア近接）。→ **Phase 1 の主戦場＝coding ＋ HLE 方向（難推論）＋ charts**、GPQA は引き分け織り込み。

### Phase 1（静的検証・agentic 不要・まずここ）

| cluster | 訓練データ（train・eval と分離） | サイズ/出典 | verifier | 狙う eval |
|---|---|---|---|---|
| **Code** | CodeContests（旧 window）＋ HARDTESTS で頑健テスト | `deepmind/code_contests`(CC-BY-4.0,~13.5K) ／ `sigcp/hardtests_problems`(47K) | secure sandbox pass@1（private+generated tests・自明テスト問題は除外） | LiveCodeBench |
| **Science MCQ** | SuperGPQA（GPQA-D とプール分離） | `m-a-p/SuperGPQA`(ODC-BY,26.5K) | exact-match 抽出 | GPQA-Diamond |
| **Hard 多領域** | OlympiadBench ＋ Omni-MATH ＋ SuperGPQA breadth（HLE は union なので portfolio） | SymPy 検証可能な深さ＋広さ | math-verify/SymPy（+TinyV fallback） | HLE（text-only 運用可） |
| **Charts（MM）** | ReachQA ＋ 自前 synthetic-matplotlib | `hewei2001/ReachQA`(MIT,20K QA/3K chart) | 数値許容（±5%）／合成は exact | CharXiv |

- **MM は charts のみ**＝唯一 M3 にルーティングしないと解けない cluster。これが「図読みを M3 へ」を学習させる。**HLE は text-only subset**で運用（vision 不要・~14% の MM 問は落とす）。

### Phase 2（agentic・重 infra・後追い）

| cluster | 訓練データ | infra | verifier | 狙う eval |
|---|---|---|---|---|
| **SWE** | SWE-Gym（実行環境付き 2,438 タスク） | per-task Docker ＋ 多ターン ＋ in-container pytest（GPU 2×A100〜32×H100） | FAIL_TO_PASS / PASS_TO_PASS | SWE-Bench Pro |
| **Terminal** | Terminal-Corpus/Nemotron-Terminal（~490K 合成）＋ `terminal-bench-rl`(~331 GRPO 済) を bootstrap | Docker terminal env | programmatic pass/fail | Terminal-Bench 2.1 |

- SWE/Terminal だけ **Docker＋multi-turn rollout** を要求。Phase 1 安定後に着手。Terminal-Bench は eval が **89 タスクのみ・train split なし**なので合成コーパスで自作。

### verifier アーキ（reward の土台・一度作って使い回す）

- **math コア**：`math-verify`/SymPy を1つ作り science 数値・HLE 定量・olympiad で共用。**主な偽陰性＝等価形**（`1/2` vs `0.5`、未簡約）→ **TinyV 風の軽量 LLM-verifier を後段 fallback** に重ね訓練崩壊を防ぐ。
- **code＝最大の reward-hacking 面**。弱いテストは誤コードを通す。CodeContests の private+generated ＋ HARDTESTS で厚くし、「public test が in-prompt 例と一致する問題は除外」。**RL は verifier の穴を必ず突く**前提で設計。
- **MCQ**：letter exact-match 抽出。**chart**：数値許容（合成は構成上 exact＝最もクリーンな reward）。
- **HLE 自由記述**は本来 GPT-4o 級 judge 依存でノイズ＋コスト大 → **訓練では SymPy 検証可能な subset を優先**、judge 採点は最小限に。

### contamination プロトコル（GRPO 前に必ず実行）

1. **temporal**（code）：旧 CodeContests/TACO で訓練し、**最新 LiveCodeBench window**（base cutoff 後公開）を eval に確保。LCB の date-stamp を利用。
2. **repo-disjoint**（SWE）：SWE-Bench Pro の **41 repo を denylist**（元々 copyleft+held-out+commercial で漏れにくい）。
3. **synthetic-by-construction**（terminal）：合成コーパスは hand-crafted 89 タスクと別物。
4. **source-disjoint**（charts）：CharXiv=arXiv 図。訓練は **business/synthetic-matplotlib のみ、arXiv 図は絶対にスクレイプしない**。
5. **全 train pool を eval set に対し text-normalized dedup** してから GRPO。

### カリキュラム順（DAG 複雑度 × タスク難度）

MCQ（最安・最クリーン信号）→ code（exec）→ olympiad/charts、を easy→hard。DAG は best-of-N/単一ルート → 逐次チェーン → 並列ツリー → 検索集約 と段階拡大（cold-start の dense format reward と同期）。

### lock 前の確認 TODO

- HARDTESTS / Terminal-Corpus / `terminal-bench-rl` の **license** 確認。PlotQA/FigureQA/AGIEval の HF id 確定。
- Fugu の HLE が full か text-only か、使った LiveCodeBench window、LiveCodeBench Pro の現サイズを公式 repo で再確認。

### pilot データ（生成済・2026-06-29）

text-only 3 cluster の pilot RL データを **LLM 生成**で作成（**OOD by construction**＝Fugu eval に非含有）。`_assets/conductor-train-pilot/pilot.jsonl`。

- **201 件**：code 71（実行検証・566 tests・weak-test 済）/ science_mcq 65（独立再解答・"常に B" hack 是正）/ hard_math 65（全件 SymPy 計算）。
- 難度 easy 59 / med 80 / hard 62、**全件 `gold_confidence=1.0`**、bad JSON 0。
- 仕様・verifier 採点規則・gold 検証法・残存リスク → [[data-spec]]（VSCode 実装 Agent への handoff）。
- charts(MM) は pilot-2、agentic（SWE/Terminal）は Phase 2。スケール時は生成→curate（[[conductor-train-task-sources-2026]]）。
- ⚠ 残存リスク：code は **reference-defined gold**（参照解が誤れば self-consistent な誤 gold）。pilot として許容、本訓練は curate へ。

---

## 📍 現在地（確定した設計）

### 構成（モデルID）

| 役割 | モデル | 価格 (in/out, per 1M) | 用途 |
|---|---|---|---|
| **Conductor** | `Qwen/Qwen3.5-4B`（ローカル vLLM/SGLang 配信） | — | 指揮者。4B・ハイブリッド（Gated DeltaNet+スパース MoE）・native 262k ctx |
| ① 高速 | `deepseek/deepseek-v4-flash` | ~$0.09–0.14 / $0.18–0.28 | 284B/13B MoE。下書き・要約・**集約（aggregator）役** |
| ② 特徴（MM/ツール） | `minimax/minimax-m3` | $0.30 / $1.20（50%オフ中） | 唯一のネイティブ・マルチモーダル。多ターン協調・ツール特化 |
| ③ 賢いA（推論/コード） | `deepseek/deepseek-v4-pro` | $0.435 / $0.87 | 1.6T/49B MoE。LiveCodeBench 93.5 / SWE Verified 80.6 を激安で |
| ④ 賢いB（長時間/ツール） | `z-ai/glm-5.2` | $1.40 / $4.40 | 長時間ワークフロー・コーディング・ツール使用に強い別ラボ枠 |

- **Claude（Opus）は外す**＝コスト優先。賢い枠2種を訓練中フル稼働させると API コストが非現実的になるため。
- **既知の懸念（許容して確定）**：①V4 Flash と ③V4 Pro が同じ DeepSeek 系で**誤りの傾向が相関**し、相補性（オーケストレーションの旨味）がやや減る。多様性を上げるなら高速枠を別ラボ（gpt-oss-120b / Nemotron 3 Super / Qwen3.5 系）に振る案があるが、**当面は DeepSeek 統一で確定**。

### Conductor の入出力フォーマット（Conductor 元論文を踏襲）

CoT のあとに**3つの等長リスト**を出力させる：

- `subtasks`：各サブタスクの自然言語の指示
- `model_id`：各サブタスクのワーカー割り当て（0–3）
- `access_list`：各サブタスクが文脈に含める過去出力のインデックス（＝通信トポロジー）

この3点だけで best-of-N / 逐次チェーン / 並列ツリー / 検索→集約まで表現できる。**パース不能な出力はフォーマット違反で弾く**。Web 検索は検索担当ワーカーに OpenRouter の `:online` 変種（or web plugin）を付ければモデル非依存で実現（M3 か V4 Flash を検索担当に）。

### 報酬設計（「早く正確に」の肝）

```
r = 0                         # フォーマット違反
r = 0.5                       # 整形式だが不正解
r = 1.0 − λ·ĉ_lat − μ·ĉ_cost  # 正解（ĉ_* は正規化した遅延/コスト, 各0〜1）
```

- **3目的（正確さ＋速度＋コスト）を採用（2026-06-29 確定）**。ペナルティ `λ·ĉ_lat + μ·ĉ_cost` は**正解時のみ**引く → 「速い/安いが間違い」が得をしない。正答性を確保した上で正解集合の中だけで速く・安い構成に寄る。GRPO のグループ相対 advantage と相性が良い。
- `λ, μ` は **0 から段階導入**（accuracy が安定してから）、小さめ（**0.05〜0.2**）で正答率が落ちない範囲で上げる。3項同時チューニングは脆いので、まず λ（速度）→ μ（コスト）の順が安全。
- `ĉ_lat` は壁時計だとノイズが大きいので**決定的プロキシ**：各モデルに遅延重み（例 **Flash=1 / M3=1.5 / V4 Pro=3 / GLM 5.2=3**）を与え、ワークフローの**クリティカルパス（並列ステップは max）**で合算し正規化。→「速いモデルを使う」だけでなく「並列化する」ことも報われる。
- `ĉ_cost` は各ワーカーの**ドル単価**（output $0.18〜$4.40）を**総和**（並列でも全 call が課金されるので max でなく sum）で正規化。レイテンシは max・コストは sum という非対称が、並列化を「速いが高い」と正しく評価させる。

### 訓練・ロールアウト設定（4B Conductor）

- grouped completions **G=8〜16** / バッチ **128〜256** / GRPO **150〜300 イテレーション** / **KL 正則化なし** / AdamW / 最大ワークフロー長 **4〜5 step**。
- 訓練データは**自動採点できる検証可能タスク**限定：数学（Math500）/ コード（LiveCodeBench）/ 科学（GPQA-Diamond）/ 一般知識（MMLU-Pro）を難易度・多様性で **500〜1000 問**。固有タスクも正解判定を自動化できれば差し込める。
- ワーカーが安価なので、このロールアウト数でも Opus/Gemini 構成より桁違いに低コスト。

### 進め方（決定：GRPO full-workflow 直行・SFT なし・2026-06-29）

routing-only スパイクも **SFT warm-start も経由せず**、base `Qwen3.5-4B` ＋ **作り込んだ system prompt** から**いきなり full-workflow を GRPO**。cold-start は SFT ではなく **in-context 仕様＋報酬整形**で跨ぐ：

1. **作り込んだ system prompt（in-context 仕様）**：各ワーカーの役割・速度・コスト、オーケストレーション primitives（単一/逐次/並列集約/検索→要約/解→検証→修正）、厳格な出力フォーマット（3 等長配列の JSON・`access_list[i]⊂{0..i-1}` で非循環・最後の subtask=最終回答）を明示。→ [[conductor_system_prompt]]（`_assets/`）。目的の優先順位（correct→fast→cheap）を prompt にも書き、報酬階層と方向を揃える。
2. **dense なフォーマット報酬**：パース部分点（配列等長・index 有効・DAG 非循環）で `r=0` 一色のグループ崩壊（std=0 → 勾配ゼロ）を回避。
3. **易→難のカリキュラム**：単純 best-of-N → 逐次チェーン → 並列ツリー → 検索集約、と DAG の複雑さを上げる。

> SFT を抜いたぶん、**system prompt の format 遵守率が立ち上がりを左右する**。instruction-tuned な 4B が初手から整形式を吐けるよう prompt を作り込み、dense format reward で補強する。GRPO は group 内 `A_i=(r_i−mean)/std` ＝ 初期に全完了とも format 違反だと std=0 で学習が死ぬため。

---

## ⏭️ 次のステップ（＝再開ポイント）

議論は Claude が以下を「一式書き起こせる」と提示した所で終了：

1. **OpenRouter 呼び出しラッパ**（並列実行 / `:online` 検索 / リトライ・フォールバック）
2. **ワークフロー実行器とパーサ**（3リスト → 実行 DAG）
3. **遅延プロキシ込みの報酬関数**
4. **GRPO ループの雛形**

> [!TODO] 次の作業（2026-06-29 更新）
> 1. **訓練タスクスイート設計**（進行中）：Fugu eval を倒す competency-cluster 設計。precise spec / dataset 出典 / verifier / カリキュラムは grounding 研究（[[fugu-eval-benchmarks-2026]] / [[conductor-train-task-sources-2026]]）後に本ノートへ追記。
> 2. **verifier 群の実装**：math(sympy/math-verify) / code(unit-test sandbox) / MC(exact-match) / chart-QA。← コード作業＝**Codex 委譲**。
> 3. **SFT warm-start データ**：合成 full-workflow（3リスト正例）で cold-start を跨ぐ。
> 4. **ワークフロー実行器**：3リスト→async DAG 実行（独立ステップは asyncio 並列＝latency proxy と整合）＋ OpenRouter ラッパ（`:online`/retry/fallback）。← **Codex**。
> 5. **GRPO ループ**：vLLM 配信＋weight sync（[[vllm-native-rl-apis-解説]] が直結）。← **Codex**。
> - 並行：OpenRouter Models API（`GET /api/v1/models`）で4 slug の `supported_parameters`（`tools`/`reasoning`/`response_format`）を実レスポンス照合。
> - 確定済み分岐：高速枠は **DeepSeek 統一**（多様性より立ち上げ優先）。
> - 方針：設計・タスク設計は本ノートで（Claude）、**コード実装は Codex 委譲**（[[.claude/skills/codex-consult/SKILL.md]]）。コード本体の正本は GitHub。

---

## 📚 背景知識（このノートで学んだこと）

### Sakana Fugu（arXiv:2606.21228 / 技術レポート）

複数のフロンティア LLM（Gemini / GPT / Claude Opus 等）を**ワーカー**として束ね、それらを指揮する**オーケストレーター（指揮者）モデル**を作る研究。新しいスケーリング軸として「モデルを巨大化する」のではなく「**オーケストレーション**」で集合知（collective intelligence）を引き出すという主張。リリースは2種：

- **Fugu**：レイテンシ重視。1クエリ＝1ワーカーを選ぶ**ルーティングのみ**。生成テキストではなく**ロジットだけ**で振り分けるので速い。
- **Fugu-Ultra**：性能重視。1クエリに**複数エージェントのワークフローを自然言語で生成**。遅いが難問の品質が上がる。

> [!WARNING] 読むときの注意
> Sakana 自身の技術レポートで**査読論文ではない**。ベースラインの多くは各プロバイダの**自己報告値**で比較条件が完全には揃っていない。「export control のリスクなしにフロンティア性能を提供できる」等の自社訴求も含む。オーケストレーター自身の**パラメータ数は非公開**。

### Trinity（arXiv:2512.04695 / ICLR 2026）— Fugu の土台

「Trinity: An Evolved LLM Coordinator」。**約0.6B の小型 LM + 約10K の軽量ヘッド**を**進化戦略（sep-CMA-ES）で最適化**。マルチターンで動作し、各ターンで選んだ LLM に **Thinker（戦略・分解）/ Worker（計算実行）/ Verifier（検証）** の3役割のいずれかを割り当てる。Fugu はこの役割割り当てを省いて「ワーカー選択のみ」に単純化し高速化したもの。

### Conductor（arXiv:2512.04388 / ICLR 2026）— Fugu-Ultra の土台

「Learning to Orchestrate Agents in Natural Language with the Conductor」。**7B を強化学習（GRPO）で訓練**。Conductor 自身が LLM で**ワークフローのステップ列を自然言語で出力**（各ステップ＝自然言語の指示・担当エージェント・他エージェント作業の可視性）。

- 訓練データ：4ドメイン（**MATH / MMLU / RLPR / LiveCodeBench V1**）から **960問**。
- 報酬：フォーマット違反=0、整形式だが不正解=0.5、正解=1.0。**200 GRPO iter / バッチ256 / KL なし**。
- ワーカープール：プロプライエタリ（Gemini-2.5-Pro / Claude-Sonnet-4 / GPT-5）＋ OSS（DeepSeek-R1-Distill-Qwen-32B / Gemma3-27B / Qwen3-32B）混在。**ランダム化したエージェントプールで訓練 → 任意のモデル集合に汎化**（＝訓練時は安価な近接モデルで代用、推論時に強いモデルへ差し替える戦略が可能）。
- 具体例：「異なる要素数が配列全体と同じになる部分配列を数えよ」に対し、Conductor は「Model 2 にアルゴリズム考案→Model 0 に Python 実装」という**役割分担の台本**を出力（自身はコードを書かない）。

### 重要な訂正：「RL」しているのは Conductor だけ

混同しやすいが、**Trinity は強化学習ではなく進化戦略（sep-CMA-ES）**。高次元・パラメータ間相関が弱い・1ステップのコストが高い条件下で sep-CMA-ES が RL・模倣学習・ランダム探索を上回る、と Trinity は主張している。

### オーケストレーター訓練の「2階層タスク」

- **土台の問題**＝ワーカーが実際に解く検証可能タスク（数学・コード・科学・一般推論、正解が一意）。
- **オーケストレーター自身のタスク**＝問題を直接解かず「どのモデルにどのサブタスクをどう振るか」（Conductor=ワークフロー台本 / Trinity=役割選択列）を出力すること。報酬はその指示を実行した最終出力の正誤から与える。
- Fugu ではここに Claude Code / Codex 由来の**実運用に近いマルチターン軌跡**が加わり、静的問題集からエージェント的ワークフローへ訓練分布が広がる。

---

## 🔗 参照

- **実装リポジトリ**：[DeL-TaiseiOzaki/conductor-rl](https://github.com/DeL-TaiseiOzaki/conductor-rl)（private）。GRPO 実装は VSCode 側。scaffold 同梱＝`prompts/`(system prompt) / `data/pilot/`(201件) / `docs/`(data-spec・reward-spec) / `configs/default.yaml`。**本ノートが design SoT、repo が実装**。
- 元チャット（全文ログ）：[[arxiv論文の解説リクエスト-aec729b9]]
- Sakana Fugu：[arXiv:2606.21228](https://arxiv.org/abs/2606.21228) / [sakana.ai/fugu](https://sakana.ai/fugu)
- Trinity：[arXiv:2512.04695](https://arxiv.org/abs/2512.04695)
- Conductor：[arXiv:2512.04388](https://arxiv.org/abs/2512.04388)
- Conductor 役モデル：[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- ワーカー：[deepseek-v4-flash](https://openrouter.ai/deepseek/deepseek-v4-flash) / [minimax-m3](https://openrouter.ai/minimax/minimax-m3) / [deepseek-v4-pro](https://openrouter.ai/deepseek/deepseek-v4-pro) / [glm-5.2](https://openrouter.ai/z-ai/glm-5.2)
- 関連 Learning：[[vibethinker-3b-論文解説]] / [[vllm-native-rl-apis-解説]]

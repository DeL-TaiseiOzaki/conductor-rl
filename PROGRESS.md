# PROGRESS

> Auto-maintained by /checkpointing. Shows the most recent 5 checkpoints (newest first).
> Full checkpoints live in `.claude/checkpoints/` (git-ignored).

## [2026-06-29-173408](.claude/checkpoints/2026-06-29-173408.md)

### 何をしたのか
- build.md の設計（Conductor=Qwen3.5-4B を GRPO で訓練し、安価 OpenRouter worker を指揮させる）を実装フェーズへ。アーキを確定：**prime-rl + `verifiers` lib + Environments Hub**、マルチノード Slurm(H100)。DESIGN.md に要件定義・技術選定・Key Decisions を記録。
- **Phase 1**（フレームワーク非依存コア）：`environments/conductor_workflow/` を起こし、parser（workflow→DAG＋graded f_fmt）/ reward（tiered）/ graders（code_exec sandbox・mcq_exact・math_verify）を実装、unit-test 化。
- **Phase 2**：workers.py（async OpenRouter・retry/fallback・semaphore・Nemotron judge）/ executor.py（DAG並列実行・f_exec・latency/cost proxy）/ `load_environment` 配線（vf.SingleTurnEnv＋Rubric）。
- **live smoke test を3クラスタで実行 → 実worker出力で grader バグ3件を発見し全修正**：math(LaTeX `\frac`偽陰性→math-verify経由)、code(フェンス未抽出→extract_code)、mcq(`A)`ラベル抽出不可→パターン追加)。end-to-end で R=1.400 を確認。
- **環境を Environments Hub に公開**：`o-taisei/conductor-workflow`(PUBLIC)。flat layout化・self-contained packaging（assets同梱）・ロード時キー要求撤去。
- **報酬を再設計**（adversarial review 反映）：絶対C_ref廃止 → **GRPOグループ相対ランクの効率報酬**（実トークンコスト＋決定的latency）、worker呼び出しキャッシュ、**ワーカー匿名化**（モデル名をConductorに見せない）。

### どういうやり取りをユーザーと行ったのか
- 着手点とフレームワークを質問 → ユーザーが「基盤+検証可能コア」「**prime-rl** を使う」を選択。続けて「**Environments Hub にenvを作り prime-rl から呼ぶ・マルチノードSlurm**」と方針を明確化。
- 各フェーズ後にコミット指示（「一旦コミットしとこう」）。Hub公開は「**publicでいい**」。
- 「Qwen未配信って？」→ モデルは存在、未起動の意と訂正。「pushしたenvがfailしてない？」→ **CI FAILEDをユーザー指摘で発見**（ロード時キー要求が原因）。
- 「CIを通す必要あるのか、無駄なAPI消費は避けたい」→ CIは必須でないと回答、Secrets追加せず赤のまま放置で合意。
- 強化学習の対象・ワークフロー設計ルール・実行順・報酬の高校生向け説明・judgeの役割（gold等価判定であって多数決でない）を順に解説。
- 追加方針：**予算上限$50**、**安いほどボーナスの報酬**、**ワーカープール匿名化**、SFTは様子見。option 2（A+B固めてGRPO直行）を選択し、Codexにadversarial review＋価格取得を指示。

### どうやったのか
- 重い実装・調査はすべて**背景サブエージェント（Opus general-purpose / codex-debugger）へ委譲**、Claudeは設計判断・検証・コミットに専念（orchestrator契約）。
- 調査2本（prime-env-authoring / prime-rl-multinode-slurm）を `.claude/docs/research/` に保存。
- 検証は毎回 Claude 自身で pytest/ruff/ty 再実行＋実APIの live smoke。OpenRouter Models API で実価格取得。Codex adversarial review（本体はtimeout、サブエージェントが代替分析）。

### 途中でどういう課題が起こったのか
- **モックが隠した実出力バグ**：3 grader が実worker出力で全滅 → live smoke で捕捉・修正（最大の学び）。
- **prime CLI の罠**：env検出は flat layout 必須（src/不可）／`--auto-bump` が `[tool.ruff] target-version` を破壊 → ruff.toml 分離で恒久対策。
- **Hub CI FAILED**：ロード時の `ensure_keys` が原因（ユーザー指摘）。
- **報酬設計の穴**：絶対C_refが最安ルートを常に最大評価する逆インセンティブ → group相対ランクへ。
- **$50予算が厳しい**：素のG=12/256/200iterはキャッシュ込みでも超過 → Phase3で G=8/batch=128/100iter/pilotのみ/強キャッシュ に縮小予定。

### 将来のアクション
- **Phase 3**：prime-rl `rl.toml`/`slurm.toml`＋sbatch を**予算ノブ込み**で作成（6 GPU vLLM+2 GPU FSDP、NFS weight sync、G=8/batch=128/100iter）。
- 訓練直前に env を Hub へ1回だけ再push（現状 v0.2.3 はコードより古い）。CI は赤許容、Secrets追加しない。
- staging：accuracy安定後に w_lat→w_cost を 0.05 から漸増、reward-hacking 監視。
- cold-start の format遵守率を監視、<50%が続けば format-only の最小SFTを緊急投入（routingはSFTしない）。
- 早期に held-out eval で Fugu 転移を測る。worker pool ランダム化は後続。

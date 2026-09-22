# Distilling MPC into Laya: telemetry-only prompts

The first experiment ([results.md](results.md)) showed that the existing
checkpoint follows whatever request is written into its prompt, so MPC guidance
made it land through engine faults, but the shield never had anything to do.
This experiment removes the requested labels from the prompt entirely. Laya
must choose the command from telemetry alone; adaptive MPC labels the training
data and, at flight time, verifies each choice by rollout.

## Setup

| | Previous checkpoint (lunar-laya) | Distilled checkpoint (this run) |
|---|---|---|
| Teacher | PD `guidance()` | Adaptive MPC (`MPCPilot("mpc")`) |
| Prompt | Telemetry **plus** requested rotation and power | Telemetry and the command currently being flown |
| Question wording | "Follow the requested tilt correction" | "Choose the tilt correction that lands safely on the target pad" |
| Rollouts | 32 seeds × 3 pads, nominal, every 8th decision | 32 seeds × 3 pads × {nominal, ×0.7, ×0.4 fault at 10 s}, every 4th decision |
| Synthetic states | 2,000 | 3,000, wider altitude range |
| Training rows | 13,022 | 49,674 |
| Validation rows | 1,928 (seeds 2000–2003) | 6,422 (seeds 2000–2003, all scenarios) |
| Epochs | 3 | 8 |
| GPUs | 3 | 4 |

The prompt carries the command the vehicle is currently flying, in the same
words as the answer choices ("Current command: tilt hold, engine half"). That
is actuator state any onboard controller can read, not the teacher's answer:
MPC's *requested* command, which the first experiment showed Laya simply copies,
is still absent. It is in the prompt because the teacher now needs it. The
adaptive MPC charges a cost for changing command between stages
(`MPCConfig.switch`, see the README), which cut the command-change rate from
77.7% to 20.6% of decisions and the stage-to-stage tilt jerk from 5.37° to 1.69°
RMS without losing a landing. That makes the flown command the single most
predictive feature of the teacher's next choice, and a student that cannot see
it is being asked to learn a function of a hidden variable. Synthetic states
have no flight behind them, so each one samples a flown command uniformly and is
labelled by the teacher given that command.

Everything else is the trainer copied from lunar-laya: `aac6fef/laya-multilingual-mlx`
at the same pinned revision, last two encoder layers plus head trainable
(24,801,793 parameters), AdamW, cosine schedule, BF16 autocast, seed 718,
all four CUDA devices on the same host, one process per GPU, batch 16 per GPU.

Files: `training/prompt.py` (questions and prompt), `training/data.py`,
`training/train.py` (copied, question import and metadata changed),
`training/pilot.py` (`DistilledPilot`), `training/verify_mlx.py`,
`tests/test_training.py`. Machine-readable results: [distillation-results.json](distillation-results.json).

## Training

| Epoch | Rank-zero train loss | Validation accuracy | Rotation | Engine |
|---|---:|---:|---:|---:|
| 1 | 0.470 | 0.853 | 0.866 | 0.840 |
| 2 | 0.405 | 0.879 | 0.886 | 0.873 |
| 3 | 0.366 | 0.889 | 0.897 | 0.880 |
| 4 | 0.335 | 0.897 | 0.910 | 0.884 |
| 5 | 0.309 | 0.893 | 0.900 | 0.887 |
| 6 | 0.294 | 0.898 | 0.910 | 0.886 |
| 7 | 0.285 | 0.900 | 0.908 | 0.892 |
| 8 | 0.286 | **0.902** | 0.913 | 0.890 |

Epoch 8 was selected. 6,216 updates in 235 seconds on four RTX PRO 4500
Blackwell GPUs (torch 2.12.1+cu129; the pinned 2.12.0 has no wheel for compute
capability 12.0). 90.2% per-question accuracy means roughly 81% of full commands
match the teacher on validation states.

The previous run of this experiment reached 80.0% with a prompt that had no
flown command in it, and with a teacher free to change command on 77.7% of
decisions. Both halves of that difference matter: the teacher is now more
predictable, and the student can see the variable the teacher is conditioned on.
Accuracy was already 85.3% after one epoch, above anything the previous run
reached in eight.

CUDA export SHA-256 `7236cde5295a03cbcaf75829d634db9586bc3278723a5423f2addd6747d9d17d`.

## Flights on held-out seeds 3000–3009, all pads

| Pilot | Scenario | Landed | Decisions | Matched MPC's request | Shield overrides |
|---|---|---:|---:|---:|---:|
| Distilled Laya, raw | nominal | 0/30 | 3,562 | 16.9% | — |
| Distilled Laya, raw | ×0.7 | 1/30 | 3,638 | 18.3% | — |
| Distilled Laya, raw | ×0.4 | 0/30 | 3,353 | 16.6% | — |
| Distilled Laya + MPC shield | nominal | **30/30** | 8,773 | 80.5% | 16.7% |
| Distilled Laya + MPC shield | ×0.7 | **30/30** | 8,588 | 85.4% | 12.2% |
| Distilled Laya + MPC shield | ×0.4 | **30/30** | 9,693 | 86.5% | 12.0% |

"Matched" counts decisions where Laya's proposal equalled MPC's own first
command on that state. "Overrides" counts proposals the shield replaced because
their best rollout cost exceeded MPC's plan (margin 0). The remaining 5–13% are
proposals that differed from MPC's choice but were accepted as no worse.

Raw outcomes are now crashes and nothing else: 30, 29 and 30 of them, plus the
one ×0.7 landing. The previous run's raw flights wandered into 6 out-of-bounds
and 5 timeouts; this student commits to a command and hits the ground with it.
That is also why raw flights are much shorter, 3.5k decisions against 13.1k.

MLX conversion: 32/32 fixture argmax agreements, maximum probability error
0.0242 against the 0.05 tolerance. Flights and conversion took 616 seconds.

## Reading it

- **Laya alone still cannot fly from telemetry, and a better student did not
  help.** 81% command agreement on in-distribution states is not enough: one
  wrong command moves the state slightly off the teacher's trajectory, the next
  state is less familiar, agreement falls to about 17%, and the flight diverges.
  Imitation learning without a corrective signal compounds errors, and raising
  validation accuracy from 80% to 90% moved raw landings from 0/90 to 1/90. The
  compounding is the problem, not the accuracy; this is the classic
  behaviour-cloning failure, reproduced here twice.
- **Laya plus the MPC shield lands every flight, now using Laya's choice more
  than four fifths of the time.** The fused pilot is neither "MPC" nor "Laya":
  Laya proposes, physics vetoes 12–17% of proposals, and the flight succeeds
  under every fault level. Against the previous checkpoint, agreement rose from
  55–62% to 80–87% and overrides fell from 30–37% to 12–17%, so the improvement
  that did nothing for the raw pilot shows up entirely as less work for the
  shield. Adaptation still comes from MPC's thrust estimate; Laya never sees it,
  and the shield's rollouts use it.
- **The shield is the safety mechanism, not a nicety.** The same weights go
  from 1/90 to 90/90. The unit test with a deliberately bad student showed the
  same shape (crash raw, land shielded with 92% overrides); the trained model
  is a much better student than that, but not a safe one.
- **Cost of the shield**: an extra beam search on every disagreement, so about
  15% of decisions pay for two plans, down from 40%.

## Next

1. **DAgger-style data.** Roll out the *shielded* pilot, label every visited
   state with MPC, retrain. The student then learns from its own mistakes'
   neighbourhoods rather than only the teacher's trajectory. This run makes the
   case stronger: better labels on the teacher's own trajectory bought 25 points
   of agreement and no raw landings, so the missing signal is off-trajectory
   states, not label quality.
2. **Calibrate and use confidence.** Fit temperatures on the validation split,
   then let the shield margin grow with Laya's confidence.
3. **Put the estimate back in the prompt**, now that the template is being
   retrained anyway, and test whether Laya can exploit it.
4. **Report the shield's own errors.** The shield accepts proposals it
   predicts are no worse; with an incomplete search and a hand-tuned cost that
   prediction can be wrong. Count landings lost to accepted-but-bad proposals
   on a harder scenario set.

## Reproduce

```bash
# Data (local, ~9 min)
uv run python -m training.data --output dist/training-data-new

# Training host, isolated environment. Pick the CUDA index for the host's cards;
# Blackwell (compute capability 12.0) needs cu128 or newer, where the closest
# build to the pinned torch==2.12.0 is 2.12.1+cu129.
uv venv --python 3.12
uv pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu129
uv pip install transformers==5.8.1 safetensors==0.7.0 'huggingface-hub>=0.34,<2' numpy \
  'laya @ git+https://github.com/NandhaKishorM/laya.git@42626c348753fbb17572a813127df2278a1ec527'

# Copy training/, lunar_mpc_laya/, lunar-laya's lunar_laya/ and the data beside it,
# then use every GPU the host has. Keep --nproc_per_node equal to the visible count.
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 -m training.train --data data --output checkpoint-new --epochs 8 --batch-size 16

# Back on the Mac
scp -r training-host:.../checkpoint-new models/lunar-mpc-laya-distilled-torch
HF_HUB_OFFLINE=1 uv run python -m training.verify_mlx \
  --source models/lunar-mpc-laya-distilled-torch --output models/lunar-mpc-laya-distilled-mlx --episodes 10
```

Checkpoints are not committed (`models/` is ignored); the SHA-256 above
identifies the trained weights.

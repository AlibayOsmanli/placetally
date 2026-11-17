# PlaceTally

Code, trained weights and stored results for PlaceTally. PlaceTally counts objects in a video that returns to rooms it has already seen. It cuts the video into fragments, links the fragments into places without looking at the question, then, for each question, reads each place once with a video-language model and adds up the per-place counts.

```
placetally/      the method: features, place finding, reading, metrics, fixed settings
scripts/         command-line scripts (below)
weights/         the trained parts: cut detector, continuation model, projection
results/         stored answers, truth and cost logs behind the paper's numbers
fragments/       stored fragment statistics for replaying the place partitions
```

The scripts that trained the three trained parts are not in this repository; Sec. B of the supplement gives their data and settings.

## Install

```sh
pip install -e .            # numpy and pillow; enough for everything on CPU
pip install -e '.[gpu]'     # torch, transformers and vLLM, to run the method on new streams
```

## Reproduce the numbers (CPU, a few minutes)

```sh
python scripts/score_answers.py    # MRA with 95% intervals for every method and split
python scripts/score_places.py     # place F1, places and exact streams on ARKit test
python scripts/return_rate.py      # the return-rate study (60 test streams, 291 questions)
python scripts/costs.py            # the cost table: build time, memory, input tokens and query time
python scripts/replay_places.py    # rebuild the place partitions (ARKit test, held-out, MultiScan)
```

`score_answers.py` covers the ARKit columns of the main results table (all but DirectMe, whose answers are not included), the long streams, the frame and token budgets and the training-set numbers. Intervals resample whole streams (5,000 draws, seed 0). `costs.py` prints the build, memory, input and query cells of the cost table, the same costs on the long streams, and the input of the budget sweep. `replay_places.py` reruns the continuation model and the best-link rule from the stored cut statistics and return similarities, and checks that every partition matches the one used in the paper. It does not redo the cuts; that needs the frames. 3RScan fragment statistics are not included (see below).

3RScan answers and fragment statistics are not included: the 3RScan terms have no clause for public redistribution, so only recording IDs and frame indices are shared. The benchmark's build scripts derive the answers from your own copy of 3RScan.

## Run on a benchmark stream (GPU)

Rebuild the stream with the benchmark's build script, with `--five-fps` for the detector frames (for example `--output rebuilt/arkit-test`; the frames go to `rebuilt/arkit-test/recordings`). Download the two checkpoints at the revisions in `placetally/settings.py`, then:

```sh
python scripts/extract_features.py --streams ../benchmark/arkit_test.json --stream arkit-test-0001 \
    --recordings rebuilt/arkit-test/recordings --encoder /path/to/siglip2-so400m-patch14-384 --output work/arkit-test-0001
python scripts/count.py --features work/arkit-test-0001 --categories sink tv_monitor \
    --reader /path/to/Qwen3.5-9B --output work/arkit-test-0001/answers.json
```

`count.py` prints the answers and saves the places and every per-place read. It needs one GPU with 40 GB. Feature extraction covers ARKit and MultiScan streams, which have 5-fps frames.

## Stored results

All files are JSON lines and use the benchmark's stream IDs.

- `results/truth.jsonl`: the answer to each question (`stream_id`, `split`, `category`, `answer`).
- `results/answers.jsonl`: one row per question and method. A null answer could not be parsed and scores 0. Methods starting with `blind:` never see the video. Methods that use the true visits or rooms are privileged: `sum over visits`, its fixed correction, `blind: rate x true rooms`, and the two `oracle:` rows.
- `results/return_rate.jsonl`: answers with visits removed from 60 test streams, at four return rates.
- `results/place_pairs.jsonl`: for each ARKit test stream and place finder, the frame pairs put in one place, in one room, and in both.
- `results/costs/`: input tokens and batched reader time per question (`questions.jsonl`), query time with nothing cached on the same 50 test questions for every method (`query_times.jsonl`), memory kept per stream (`memory.jsonl`), construction time of the 9 timed streams (`construction.jsonl`) and the budget sweep (`budgets.jsonl`). Times were measured on one A100-40GB GPU. `other_methods.json` holds the totals for VideoAgent, CountVid, DirectMe and Cambrian-S's memory, which were measured with their own code.

Answers of the other video models (Cambrian-S, VideoAgent, CountVid) were produced with their released code. The whole-video reading prompt is given in the supplement.

## Licence

MIT (see `LICENSE`). The datasets and pretrained checkpoints keep their own terms.

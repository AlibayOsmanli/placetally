# RevisitCount

RevisitCount measures counting of distinct objects in videos that return to rooms already seen. A stream joins recordings of real rooms into one long walk, and some rooms are visited again, sometimes through a different recording of the same room. A question asks how many different objects of one category the whole stream shows. Adding up the per-visit counts over-counts every object seen on a return.

The streams are built from ARKitScenes, 3RScan and MultiScan. This folder holds the stream definitions, the questions and answers, and scripts that rebuild the frames from your own copy of each dataset. It holds no images, video, depth or meshes.

| File | Streams | Questions | Use |
|---|---|---|---|
| `arkit_training.json` | 210 | 779 | fitting (two training sets; see below) |
| `arkit_test.json` | 109 | 467 | test |
| `arkit_held_out.json` | 90 | 403 | held-out test, evaluated once |
| `arkit_long.json` | 58 | 270 | streams of about 30 and 60 minutes |
| `3rscan_training.json` | 22 | rebuilt locally | fitting |
| `3rscan_test.json` | 150 | rebuilt locally | test |
| `multiscan.json` | 33 | none | place finding only |

`FORMAT.md` describes the fields. `stats.json` holds the statistics reported in the supplement. `checksums.sha256` lists the SHA-256 of every file.

## Splits

The training file holds two sets. The first has 65 streams, of which 48 were used for fitting; the other 17 lack 5-fps frames for one recording (`fit_cohort` is false). The second has 145 streams, all used. The 193 fitting streams have 694 questions, marked `scored`.

The test file has 467 questions on 105 of its 109 streams. Seven questions were lost to a request error before scoring, so test scores use 460 questions on 103 streams; the seven are marked `scored: false`. Four test streams have no question but still count for place finding.

No recording or room of a test or held-out stream appears in a training stream.

3RScan and MultiScan files give the recordings, frame indices and order. The 3RScan terms have no clause for public redistribution, so the 3RScan files carry no room identities, questions or answers. The build script derives them from your copy. MultiScan streams have room identities and are used to score place finding; they have no questions.

## Rebuilding the frames

Install `build/requirements.txt`, download the source data unchanged, and run the build script once per file, into an empty folder:

```sh
python build/rebuild.py --dataset arkit --source /path/to/ARKitScenes --release arkit_test.json --output rebuilt/arkit-test
python build/rebuild.py --dataset 3rscan --source /path/to/3RScan --release 3rscan_test.json --output rebuilt/3rscan-test
python build/rebuild.py --dataset multiscan --source /path/to/MultiScan --release multiscan.json --output rebuilt/multiscan
```

`--stream arkit-test-0001` rebuilds one stream. `--five-fps` also writes the 5-fps frames that PlaceTally's cut detector reads (ARKit and MultiScan only). The script writes one archive per recording to `<output>/recordings`, holding the JPEG frames and the visible object boxes in each frame, and then recomputes the answers. For ARKit it stops if any answer differs from the released one. For 3RScan it writes the answers and room identities to `locally_derived_answers.json`. It runs on CPU.

Expected source layouts:

- ARKitScenes ([download](https://github.com/apple/ARKitScenes)): `3dod/Training/<video>/` and `raw/Training/<video>/`, with the high-resolution frames and their intrinsics.
- 3RScan ([access form](https://github.com/WaldJohannaU/3RScan)): `3RScan.json` and `scans/<recording>/` with `sequence.zip`, the annotated mesh and the semantic segmentation.
- MultiScan ([download](https://huggingface.co/datasets/3dlg-hcvc/MultiScan)): `<recording>/<recording>.mp4`, `.jsonl`, `.ply`, `.annotations.json` and `.align.json`.

## How the answers are made

ARKitScenes and MultiScan frames are taken at 1 fps; 3RScan streams use its own keyframes. An object counts as seen in a frame when enough of its annotated box is visible: for ARKitScenes, sample points on the camera-facing faces of the box are checked against the sensor depth; for 3RScan and MultiScan, the annotated mesh is rendered with a depth buffer. An object counts as seen in a visit when its visible box covers at least 0.5% of the image in at least two frames. When a room returns through another recording, matched object instances are linked across the two recordings (`object_identity_links`); categories whose matches are ambiguous are left out for that stream. A question is kept only when its answer is at least 1 and differs from the sum of the per-visit counts. Sec. A of the supplement gives the details.

Frame indices refer to the recording's sampled frames, not to native video frames. Times are in seconds, one per frame; a 2-second blank follows each visit, so a visit starts 3 s after the previous visit's last frame. Frames are stored as recorded. Rotate them upright with the visit's `orientation` label (degrees counter-clockwise: Up 0, Left -90, Down 180, Right 90). There is one label per recording, so a few frames can stay sideways.

## Licence

The stream definitions and annotations are released under CC BY-NC 4.0 (`LICENSE`). The source datasets keep their own terms: ARKitScenes under Apple's licence for non-commercial use, 3RScan under its terms of use for non-commercial research, and MultiScan under CC BY-NC 4.0. If you use RevisitCount, please also cite:

- G. Baruch et al. ARKitScenes: A Diverse Real-World Dataset for 3D Indoor Scene Understanding Using Mobile RGB-D Data. NeurIPS Datasets and Benchmarks, 2021.
- J. Wald et al. RIO: 3D Object Instance Re-Localization in Changing Indoor Environments. ICCV, 2019.
- Y. Mao et al. MultiScan: Scalable RGBD Scanning for 3D Environments with Articulated Objects. NeurIPS, 2022.

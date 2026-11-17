# PlaceTally: A Test-Time Memory Model for Constructing Place Graphs

Alibay Osmanli and Shaogang Gong, Queen Mary University of London

**[Project page](https://alibayosmanli.github.io/placetally)** · [Paper (PDF)](https://alibayosmanli.github.io/placetally/paper.pdf) · [Supplementary material (PDF)](https://alibayosmanli.github.io/placetally/supplement.pdf)

PlaceTally counts distinct objects in a video that returns to rooms it has already seen. Before any question is asked, it builds a place graph for the video: each fragment joins the place of its best earlier match or opens a new place. A frozen vision-language model then counts each place once, and the per-place counts are added up. RevisitCount is the benchmark of real room re-recordings used in the paper.

| Folder | Contents |
|---|---|
| `code/` | PlaceTally: the method, its trained weights, the stored answers behind the paper's numbers, and scripts that recompute them ([code/README.md](code/README.md)) |
| `benchmark/` | RevisitCount: stream definitions, questions and answers, and scripts that rebuild the frames from the source datasets ([benchmark/README.md](benchmark/README.md)) |
| `docs/` | the project page, with the paper and the supplementary material |

## Reproduce the paper's ARKit numbers on a CPU

```sh
cd code
pip install -e .
python scripts/score_answers.py
```

`code/README.md` lists the other scripts (place finding, the return-rate study, the cost table) and explains how to run the method on a benchmark stream with a GPU.

## Supplementary material

The latest supplementary material is `docs/supplement.pdf`. Each version is also kept under a dated name, starting with `docs/supplement_v1_2026-11-17.pdf`. Later versions keep the section letters A to H and their numbering, so references in the paper such as "Supp. C" keep pointing to the right place.

## Licence

The code is under the MIT licence (`code/LICENSE`), and the benchmark's stream definitions and annotations are under CC BY-NC 4.0 (`benchmark/LICENSE`). ARKitScenes, 3RScan and MultiScan keep their own terms; `code/` and `benchmark/` hold no images, video or meshes from them.

## Citation

```bibtex
@misc{osmanli2026placetally,
  title  = {PlaceTally: A Test-Time Memory Model for Constructing Place Graphs},
  author = {Osmanli, Alibay and Gong, Shaogang},
  year   = {2026},
  url    = {https://alibayosmanli.github.io/placetally}
}
```

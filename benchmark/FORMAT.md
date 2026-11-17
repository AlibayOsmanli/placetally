# File format

Each file is a JSON object with `format_version` (1) and a list `streams`. Stream IDs are numbered in order within each file, for example `arkit-test-0001`.

## Stream

| Field | Meaning |
|---|---|
| `stream_id` | Stream ID. |
| `split` | `training`, `test` or `held-out`. Long streams are test streams. |
| `visits` | The visits in stream order (below). |
| `duration_s` | Length of the stream in seconds, ending 3 s after the last frame. |
| `distinct_rooms` | Number of different rooms in the stream. |
| `planned_rooms`, `planned_returns` | Rooms and returns the stream was built with. |
| `ends_in_new_room` | Whether a visit to one more new room follows the returns. |
| `look_alike_room` | Null, or a pair of recordings of different rooms that look alike, with their similarity. |
| `object_identity_links` | Object instances linked across two recordings of the same room, as `"recording:object": "recording:object"`. A linked pair is one object. |
| `excluded_categories` | Categories left out of this stream because objects could not be matched across recordings. |
| `questions` | The questions (below). |
| `training_set` | Training streams only: 1 or 2. |
| `fit_cohort`, `not_fit_reason` | Training streams only: whether the stream was used for fitting, and why not. |
| `length_group` | Long streams only: 30 or 60 (minutes, roughly). |
| `source_frame_stride` | MultiScan only: the stride of the sampled frame table. |

## Visit

| Field | Meaning |
|---|---|
| `recording_id` | Recording ID in the source dataset. A return can use another recording of the same room. |
| `room_id` | Room ID. Two visits with the same `room_id` show the same room. |
| `frame_ranges` | Frames of the recording shown in this visit, as ranges of the recording's sampled frame indices. Each range has `first`, `count` and `step` (negative when the visit runs backwards), and `time_first` and `time_step` for the frame times in the stream. |
| `reversed` | Whether the visit plays the recording backwards. |
| `orientation` | `Up`, `Left`, `Down` or `Right`: how to rotate the recording's frames upright. |

## Question

| Field | Meaning |
|---|---|
| `category` | One of the 14 categories, e.g. `chair` or `tv_monitor` (TV or computer monitor). |
| `answer` | Number of different objects of the category seen in the whole stream. |
| `per_visit_counts` | Number seen in each visit. Their sum counts an object again on every return. |
| `scored` | Whether the question is in the scored set: the fitting streams for training, and all but seven test questions. |
| `unscored_reason` | Why a question is not scored. |

3RScan files contain only `stream_id` and, per visit, `recording_id` and `frame_ranges` (`first`, `count`, `step`). Room identities, orientations, object links and questions are derived from your copy of 3RScan by `build/rebuild.py`. MultiScan files have no `questions` and no object fields.

"""Fixed settings. All were set before the test streams were built."""

# Place finding
CUT_THRESHOLD = 3.792  # detector log-odds above which a peak is a cut
PEAK_GAP = 10  # minimum distance between cuts, in 5-fps frame pairs (2 s)
TAU = 0.8637  # return threshold on SigLIP 2 chamfer similarity

# Reading
FRAMES_PER_PLACE = 128
CANVAS = 448
TAU_GATE = 0.6487419423953118
COUNT_TOKENS = 8
GATE_TOP_LOGPROBS = 20

READER = "Qwen/Qwen3.5-9B"
READER_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ENCODER = "google/siglip2-so400m-patch14-384"
ENCODER_REVISION = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"

"""Reading a place with the video-language model: prompts, frame choice, the gate and the count."""

import math
import re

import numpy as np
from PIL import Image

from . import settings
from .features import upright

CATEGORY_NAMES = {
    "bathtub": "bathtub",
    "bed": "bed",
    "cabinet": "cabinet",
    "chair": "chair",
    "dishwasher": "dishwasher",
    "fireplace": "fireplace",
    "oven": "oven",
    "refrigerator": "refrigerator",
    "shelf": "shelf",
    "sink": "sink",
    "sofa": "sofa",
    "stool": "stool",
    "stove": "stove",
    "table": "table",
    "toilet": "toilet",
    "tv_monitor": "TV or computer monitor",
    "washer": "washing machine",
}

INTRO = "This clip is one walk through a single room or space, filmed with a handheld camera."
COUNT_QUESTION = (
    "How many different {plural} are in this space? Count each one once, even if it appears in several frames. "
    "Answer with a single number. If there are none, answer 0."
)


def plural(category):
    name = CATEGORY_NAMES[category]
    return "shelves" if name == "shelf" else name + "s"


def count_question(category):
    return COUNT_QUESTION.format(plural=plural(category))


def presence_question(category):
    name = CATEGORY_NAMES[category]
    article = "an" if name[0].lower() in "aeiou" else "a"
    return f"Is there {article} {name} in this space? Answer yes or no."


def sample_frames(n, k=settings.FRAMES_PER_PLACE):
    """Indices of at most k frames spread evenly over n; the last is repeated if the count is odd."""
    chosen = np.arange(n)
    if n > k:
        chosen = np.unique(np.linspace(0, n - 1, k).round().astype(int))
    return np.concatenate([chosen, chosen[-1:]]) if len(chosen) % 2 else chosen


def letterbox(path, orientation, size=settings.CANVAS):
    """The frame made upright, scaled to fit and centred on a black square."""
    im = upright(Image.open(path).convert("RGB"), orientation)
    scale = size / max(im.size)
    im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.BICUBIC)
    out = Image.new("RGB", (size, size))
    out.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return np.asarray(out)


def make_video(frames, times):
    """A video input for vLLM: the frames with the stream's own timestamps, already sampled."""
    metadata = {
        "total_num_frames": len(frames),
        "fps": 1.0,
        "frames_indices": [float(t) for t in times],
        "height": int(frames.shape[1]),
        "width": int(frames.shape[2]),
        "duration": float(times[-1] - times[0] + 1),
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    return frames, metadata


def first_integer(text):
    match = re.search(r"\d+", text.replace(",", ""))
    return int(match.group()) if match else None


def gated_count(count, p_yes):
    return (count or 0) if p_yes >= settings.TAU_GATE else 0


class Reader:
    """Qwen3.5-9B served with vLLM, greedy decoding."""

    def __init__(self, checkpoint):
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self.processor = AutoProcessor.from_pretrained(checkpoint)
        tokenizer = self.processor.tokenizer
        self.llm = LLM(
            model=checkpoint,
            max_model_len=32768,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
            gpu_memory_utilization=0.90,
            seed=0,
            tensor_parallel_size=1,
            limit_mm_per_prompt={"image": 0, "video": 1},
            mm_processor_kwargs={
                "size": {"longest_edge": settings.FRAMES_PER_PLACE * settings.CANVAS**2, "shortest_edge": 4096}
            },
            mm_processor_cache_gb=4.0,
        )
        self.count_params = SamplingParams(temperature=0.0, max_tokens=settings.COUNT_TOKENS, seed=0)
        self.gate_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=settings.GATE_TOP_LOGPROBS, seed=0)
        self.yes_ids = self._single_token_ids(tokenizer, "yes")
        self.no_ids = self._single_token_ids(tokenizer, "no")

    @staticmethod
    def _single_token_ids(tokenizer, word):
        """Token ids of the spellings word, " word", Word and " Word" that are a single token."""
        ids = set()
        for spelling in (word, " " + word, word.capitalize(), " " + word.capitalize()):
            encoded = tokenizer.encode(spelling, add_special_tokens=False)
            if len(encoded) == 1:
                ids.add(encoded[0])
        return ids

    def _prompt(self, question):
        content = [{"type": "text", "text": INTRO}, {"type": "video"}, {"type": "text", "text": "\n" + question}]
        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def _p_yes(self, output):
        logprobs = output.outputs[0].logprobs[0] if output.outputs[0].logprobs else {}
        p_yes = sum(math.exp(logprobs[t].logprob) for t in self.yes_ids if t in logprobs)
        p_no = sum(math.exp(logprobs[t].logprob) for t in self.no_ids if t in logprobs)
        return p_yes / (p_yes + p_no) if p_yes + p_no > 0 else 0.5

    def read(self, video, category, video_id):
        """Count and gate reads of one place for one category. Returns (count, p_yes, input tokens)."""
        requests = [
            {
                "prompt": self._prompt(question),
                "multi_modal_data": {"video": [video]},
                "multi_modal_uuids": {"video": [video_id]},
            }
            for question in (count_question(category), presence_question(category))
        ]
        # As in the paper's runs: the first request alone, then the rest.
        count_out = self.llm.generate(requests[:1], [self.count_params], use_tqdm=False)[0]
        gate_out = self.llm.generate(requests[1:], [self.gate_params], use_tqdm=False)[0]
        count = first_integer(count_out.outputs[0].text.strip())
        tokens = len(count_out.prompt_token_ids) + len(gate_out.prompt_token_ids)
        return count, self._p_yes(gate_out), tokens

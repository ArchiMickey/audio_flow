# Local copy of the ArchiTTS eval WER/SIM entry points used by audio_flow.

from __future__ import annotations

import os
import string
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from audio_flow.eval.ecapa_tdnn import ECAPA_TDNN_SMALL


def load_asr_model(lang: str, ckpt_dir: str = "", **kwargs):
    if lang == "zh":
        from funasr import AutoModel

        return AutoModel(
            model=os.path.join(ckpt_dir, "paraformer-zh"),
            disable_update=True,
            **kwargs,
        )
    if lang == "en":
        from faster_whisper import WhisperModel

        model_size = "large-v3" if ckpt_dir == "" else ckpt_dir
        return WhisperModel(model_size, device="cuda", compute_type="float16", **kwargs)
    raise NotImplementedError(
        "lang support only 'zh' (funasr paraformer-zh), 'en' (faster-whisper-large-v3), for now."
    )


def run_asr_wer(args):
    rank, lang, test_set, ckpt_dir = args

    if lang == "zh":
        import zhconv

        torch.cuda.set_device(rank)
    elif lang == "en":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    else:
        raise NotImplementedError(
            "lang support only 'zh' (funasr paraformer-zh), 'en' (faster-whisper-large-v3), for now."
        )

    asr_model = load_asr_model(lang, ckpt_dir=ckpt_dir)

    from jiwer import process_words
    from zhon.hanzi import punctuation

    punctuation_all = punctuation + string.punctuation
    wer_results = []

    for gen_wav, _prompt_wav, truth in tqdm(test_set):
        if lang == "zh":
            res = asr_model.generate(input=gen_wav, batch_size_s=300, disable_pbar=True)
            hypo = zhconv.convert(res[0]["text"], "zh-cn")
        elif lang == "en":
            segments, _ = asr_model.transcribe(gen_wav, beam_size=5, language="en")
            hypo = ""
            for segment in segments:
                hypo = hypo + " " + segment.text

        raw_truth = truth
        raw_hypo = hypo

        for item in punctuation_all:
            truth = truth.replace(item, "")
            hypo = hypo.replace(item, "")

        truth = truth.replace("  ", " ")
        hypo = hypo.replace("  ", " ")

        if lang == "zh":
            truth = " ".join([item for item in truth])
            hypo = " ".join([item for item in hypo])
        elif lang == "en":
            truth = truth.lower()
            hypo = hypo.lower()

        measures = process_words(truth, hypo)
        wer_results.append(
            {
                "wav": Path(gen_wav).stem,
                "truth": raw_truth,
                "hypo": raw_hypo,
                "wer": measures.wer,
            }
        )

    return wer_results


def run_sim(args):
    rank, test_set, ckpt_dir = args
    device = f"cuda:{rank}"

    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state_dict = torch.load(ckpt_dir, weights_only=True, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict["model"], strict=False)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda(device)
    model.eval()

    sim_results = []
    for gen_wav, prompt_wav, _truth in tqdm(test_set):
        wav1, sr1 = torchaudio.load(gen_wav)
        wav2, sr2 = torchaudio.load(prompt_wav)

        wav1 = torchaudio.transforms.Resample(orig_freq=sr1, new_freq=16000)(wav1)
        wav2 = torchaudio.transforms.Resample(orig_freq=sr2, new_freq=16000)(wav2)

        if use_gpu:
            wav1 = wav1.cuda(device)
            wav2 = wav2.cuda(device)
        with torch.no_grad():
            emb1 = model(wav1)
            emb2 = model(wav2)

        sim_results.append(
            {
                "wav": Path(gen_wav).stem,
                "sim": F.cosine_similarity(emb1, emb2)[0].item(),
            }
        )

    return sim_results

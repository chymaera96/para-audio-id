"""Resolve the frozen-FSQ PSI experiment using the established 25K recipe."""
from copy import deepcopy
import math

import torch


def resolve_fsq_psi_config(config, *, decoder, schedule, selected_codebooks,
                           distillation_weight, checkpoint):
    from .fsq import FrozenFSQTokenizer
    from .profiles import decoder_profile, resolve_training_config

    saved = None
    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        profile = saved["resolved_training_profile"]
        decoder = decoder or profile["decoder"]["name"]
        schedule = schedule or profile["schedule"]["name"]
        if distillation_weight is None:
            distillation_weight = profile["distillation"]["maximum_weight"]
    base = deepcopy(config)
    selected_decoder = decoder or base.get("model", {}).get("decoder", "small")
    if selected_decoder not in ("small", "medium"):
        raise ValueError("FSQ PSI supports small or medium decoders")
    tokenizer_cfg = deepcopy(base["tokenizer"])
    policy = tokenizer_cfg.get("execution_policy", "inherit")
    frozen = FrozenFSQTokenizer(tokenizer_cfg["checkpoint"], device="cpu", execution_policy=policy)
    groups = frozen.selected_codebooks
    rate = frozen.model.frame_rate
    if selected_codebooks is not None and selected_codebooks != groups:
        raise ValueError("--codebooks must match the frozen FSQ group count")
    if tokenizer_cfg.get("selected_codebooks", groups) != groups:
        raise ValueError("Configured FSQ group count differs from frozen snapshot")
    if frozen.vocabulary.codebook_size != 1024 or frozen.sample_rate != 24000:
        raise ValueError("FSQ PSI expects 1024-symbol groups at 24 kHz")
    frames = 2 * rate
    if not math.isfinite(frames) or frames <= 0 or not frames.is_integer():
        raise ValueError("FSQ must produce an integer number of frames per two-second crop")
    id_weight = 4.0 * groups * rate / 25
    # Derive the existing exposure, augmentation, LR, and decoder recipe first.
    # This recursive call uses MuQ configuration only to resolve that recipe;
    # no MuQ model is instantiated or downloaded.
    base["tokenizer"] = {"selected_codebooks": 2, "sample_rate": 24000}
    resolved = resolve_training_config(base, decoder="small", schedule=schedule,
                                      selected_codebooks=2,
                                      distillation_weight=distillation_weight)
    tokenizer_cfg.update(sample_rate=24000, selected_codebooks=groups,
                         checkpoint=frozen.spec.model_name,
                         checkpoint_sha256=frozen.spec.revision)
    resolved["tokenizer"] = tokenizer_cfg
    resolved["train"]["id_digit_weight"] = id_weight
    resolved["resolved_query_profile"] = {"selected_codebooks": groups, "id_digit_weight": id_weight}
    profile = resolved["resolved_training_profile"]
    profile["decoder"] = decoder_profile(selected_decoder)
    resolved["model"].update({k: v for k, v in profile["decoder"].items() if k != "name"})
    resolved["model"]["decoder"] = selected_decoder
    legacy = groups == 1 and rate == 25 and policy == "inherit"
    profile["variant"] = "cnn-fsq-single-group-25k-v1" if legacy else "cnn-fsq-25k-v2"
    profile["training_tracks_manifest"] = config["data"].get(
        "training_tracks_manifest", profile["training_tracks_manifest"])
    resolved["data"]["training_tracks_manifest"] = profile["training_tracks_manifest"]
    profile["tokenizer_fingerprint"] = frozen.spec.fingerprint
    protocol = "cnn_fsq_psi_v1"
    for entry in (profile["schedule"], profile["distillation"],
                  resolved["train"]["schedule"], resolved["train"]["distillation"]):
        entry["protocol"] = protocol
        if "loss_protocol" in entry:
            entry["loss_protocol"] = protocol
    if saved is not None:
        if saved.get("resolved_training_profile") != profile:
            raise ValueError("Resume FSQ PSI profile differs from the saved checkpoint")
        if saved.get("tokenizer_fingerprint") != frozen.spec.fingerprint:
            raise ValueError("Resume tokenizer weights have changed")
    return resolved

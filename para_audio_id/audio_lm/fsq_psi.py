"""Resolve the frozen-FSQ PSI experiment using the established 25K recipe."""
from copy import deepcopy

import torch


def resolve_fsq_psi_config(config, *, decoder, schedule, selected_codebooks,
                           distillation_weight, checkpoint):
    from .fsq import FrozenFSQTokenizer
    from .profiles import resolve_training_config

    saved = None
    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        profile = saved["resolved_training_profile"]
        decoder = decoder or profile["decoder"]["name"]
        schedule = schedule or profile["schedule"]["name"]
        if distillation_weight is None:
            distillation_weight = profile["distillation"]["maximum_weight"]
    if selected_codebooks not in (None, 1):
        raise ValueError("This pilot uses one FSQ group; omit --codebooks")
    base = deepcopy(config)
    tokenizer_cfg = deepcopy(base["tokenizer"])
    frozen = FrozenFSQTokenizer(tokenizer_cfg["checkpoint"], device="cpu")
    if frozen.selected_codebooks != 1 or frozen.vocabulary.codebook_size != 1024:
        raise ValueError("PSI pilot expects one 1024-symbol FSQ group")
    if frozen.sample_rate != 24000 or frozen.model.frame_rate != 25:
        raise ValueError("PSI pilot expects 24 kHz audio and 25 latent frames/s")
    # Derive the existing exposure, augmentation, LR, and decoder recipe first.
    # This recursive call uses MuQ configuration only to resolve that recipe;
    # no MuQ model is instantiated or downloaded.
    base["tokenizer"] = {"selected_codebooks": 2, "sample_rate": 24000}
    resolved = resolve_training_config(base, decoder=decoder, schedule=schedule,
                                      selected_codebooks=2,
                                      distillation_weight=distillation_weight)
    tokenizer_cfg.update(sample_rate=24000, selected_codebooks=1,
                         checkpoint=frozen.spec.model_name,
                         checkpoint_sha256=frozen.spec.revision)
    resolved["tokenizer"] = tokenizer_cfg
    resolved["train"]["id_digit_weight"] = 4.0
    resolved["resolved_query_profile"] = {"selected_codebooks": 1, "id_digit_weight": 4.0}
    profile = resolved["resolved_training_profile"]
    profile["variant"] = "cnn-fsq-single-group-25k-v1"
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

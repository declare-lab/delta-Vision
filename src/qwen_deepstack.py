"""Project-wide DeepStack-off policy for Qwen training and inference."""


def _reject_deepstack(*args, **kwargs):
    raise AssertionError("DeepStack execution is disabled project-wide")


def disable_qwen_deepstack(model):
    """Disable auxiliary vision mergers and language injection before any forward.

    Accept either the outer generation model or its multimodal backbone. Keep
    checkpoint parameters/configuration intact so loading and saving stay compatible.
    The main vision merger is unchanged. Qwen3.5 already has no DeepStack branches.
    """
    backbone = model if hasattr(model, "visual") else model.model
    visual = backbone.visual
    language = backbone.language_model
    visual.deepstack_visual_indexes = []
    if hasattr(language, "_deepstack_process"):
        language._deepstack_process = _reject_deepstack
    if not hasattr(backbone, "_deepstack_guard_handles"):
        backbone._deepstack_guard_handles = [
            module.register_forward_pre_hook(_reject_deepstack)
            for module in getattr(visual, "deepstack_merger_list", ())
        ]
    backbone._benchmark_deepstack = "off"
    model._benchmark_deepstack = "off"
    return model

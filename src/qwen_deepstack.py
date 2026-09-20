"""Explicit DeepStack-off policy for Qwen efficiency comparisons."""


def disable_qwen_deepstack(model):
    """Skip vision side mergers and reject any language-side injection.

    Apply before warming/capturing graphs. The main vision merger remains active.
    """
    if getattr(model, "_benchmark_deepstack", None) == "off":
        return
    visual = model.model.visual
    language = model.model.language_model
    visual.deepstack_visual_indexes = []

    def reject(*args, **kwargs):
        raise AssertionError("DeepStack executed in a DeepStack-off benchmark")

    language._deepstack_process = reject
    model._deepstack_guard_handles = [module.register_forward_pre_hook(reject)
        for module in visual.deepstack_merger_list]
    model._benchmark_deepstack = "off"

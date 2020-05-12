
def clip_grad_by_value(value):
    import torch

    def impl(params):
        torch.nn.utils.clip_grad_value_(params, value)

    return impl


def clip_grad_by_norm(norm):
    import torch

    def impl(params):
        torch.nn.utils.clip_grad_norm_(params, norm)

    return impl

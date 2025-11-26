import torch, random
from torch.autograd import grad
from typing import Callable, Optional
from torch import nn

def initialize_weights(model):
    model.w0_dict = dict()
    for param in model.named_parameters():
        model.w0_dict[param[0]] = param[1].clone().detach()
    print("initialize model weights.")


def compute_information_fast(model, loss, beta=1e-2,
                             param_frac=1.0, seed=None,
                             no_bp=False, grad_scale=1e-5):
    """
    Compute Fisher-like information penalty using (Δw^T g)^2,
    but ONLY for a randomly sampled subset of parameter tensors.
    """

    weight_names = [n for n, _ in model.named_parameters() if "35" in n]
    if seed is not None:
        random.seed(seed)
    if param_frac < 1.0:
        k = max(1, int(len(weight_names) * param_frac))
        keep = set(random.sample(weight_names, k))
    else:
        keep = set(weight_names)


    delta_w_dict = {}
    for name, p in model.named_parameters():
        if name in keep:
            w0 = model.w0_dict[name].to(p.device, dtype=p.dtype)
            delta_w_dict[name] = p - w0


    keep_names_sorted = []
    keep_params = []
    for name, p in model.named_parameters():
        if name in keep:
            keep_names_sorted.append(name)
            keep_params.append(p)


    grads_subset = grad(loss,
                        keep_params,
                        create_graph=not no_bp,
                        allow_unused=True)


    terms = []
    for name, g, in zip(keep_names_sorted, grads_subset):
        if g is None:
            # This kept parameter did not influence the loss; skip it
            continue
        gflat = g.to(p.dtype).flatten().detach() * grad_scale
        dw = delta_w_dict[name].flatten()
        info_ = (dw * gflat).sum()
        info_ = (info_ ** 2).to(p.dtype)
        terms.append(info_ if no_bp else info_)

    if len(terms) == 0:
        return torch.zeros((), device=next(model.parameters()).device)

    info_decay = beta * torch.stack(terms).sum()

    
    # for p in model.parameters():
    #     p.grad = None

    return info_decay



@torch.no_grad()
def compute_information_fd(
    model: nn.Module,
    eval_loss_vec_fn: Callable[[], torch.Tensor],
    beta: float = 1e-2,
    eps: float = 1e-3,
    param_frac: float = 1.0,
    seed: Optional[int] = None,
    name_filter: str = "35",
) -> torch.Tensor:
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    weight_names = [n for n, p in named_params if name_filter in n]

    if len(weight_names) == 0:
        loss_vec = eval_loss_vec_fn()
        return torch.zeros_like(loss_vec)

    if seed is not None:
        random.seed(seed)

    if 0.0 < param_frac < 1.0:
        k = max(1, int(len(weight_names) * param_frac))
        keep = set(random.sample(weight_names, k))
    else:
        keep = set(weight_names)

    name_to_param = dict(named_params)
    keep_params = {name: name_to_param[name] for name in keep}

    if len(keep_params) == 0:
        loss_vec = eval_loss_vec_fn()
        return torch.zeros_like(loss_vec)


    orig_data = {}
    delta_dict = {}

    for name, p in keep_params.items():

        orig = p.data
        orig_data[name] = orig.clone()


        w0 = model.w0_dict[name].to(device=orig.device, dtype=orig.dtype)
        delta = orig - w0
        delta_dict[name] = delta


    def apply_offsets(sign: float) -> None:
        for name, p in keep_params.items():
            base = orig_data[name]
            delta = delta_dict[name]
            p.data = base + sign * eps * delta


    apply_offsets(+1.0)
    loss_plus = eval_loss_vec_fn()


    apply_offsets(-1.0)
    loss_minus = eval_loss_vec_fn()


    for name, p in keep_params.items():
        p.data.copy_(orig_data[name])


    directional = (loss_plus - loss_minus) / (2.0 * eps)


    info_vec = beta * directional.pow(2)

    return info_vec

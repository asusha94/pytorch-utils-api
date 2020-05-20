
def metrics_ema(running_metrics, metrics, a=0.1, step=1):
    for k, v in metrics.items():
        if k not in running_metrics:
            running_metrics[k] = (1 - a) * v
        else:
            running_metrics[k] = (1 - a) * v + a * running_metrics[k]

        running_metrics[k] /= (1 - a**step)


def metrics_accumulate(running_metrics, metrics, factor=1.):
    for k, v in metrics.items():
        try:
            if k not in running_metrics:
                running_metrics[k] = v * factor
            else:
                running_metrics[k] += v * factor
        except Exception as ex:
            print('Metrics mergin error:', k, repr(ex))


def evaluate_batch(model, batch, *, step_func, calc_metrics, ret_result=False):
    from . import model as model_utils

    if not isinstance(step_func, model_utils._ForwardStepWrapper):
        step_func = model_utils._ForwardStepWrapper(step_func)

    if not isinstance(calc_metrics, model_utils._CalcMetricsWrapper):
        calc_metrics = model_utils._CalcMetricsWrapper(calc_metrics)

    result = step_func(model, batch)

    metrics = calc_metrics(model, batch, result)
    assert isinstance(metrics, dict), 'Must be a dictionary'

    if ret_result:
        return metrics, result
    else:
        return metrics


def _metrics_mean(running_metrics, steps):
    return {k: (v / steps) for k, v in running_metrics.items()}


def evaluate(model, dataset, *, step_func, calc_metrics, ret_last_batch=False, device=None):
    import torch
    from . import model as model_utils

    if not isinstance(step_func, model_utils._ForwardStepWrapper):
        step_func = model_utils._ForwardStepWrapper(step_func)

    if not isinstance(calc_metrics, model_utils._CalcMetricsWrapper):
        calc_metrics = model_utils._CalcMetricsWrapper(calc_metrics)

    if device is None:
        device = next(model.parameters()).device

    with model_utils.evaluating(model):
        with torch.no_grad():
            i = 1
            running_metrics = dict()
            for i, batch in enumerate(dataset, 1):
                batch = [item.to(device) for item in batch]

                metrics, result = evaluate_batch(model, batch,
                                                 step_func=step_func,
                                                 calc_metrics=calc_metrics,
                                                 ret_result=True)

                metrics_accumulate(running_metrics, metrics)
            else:
                running_metrics = _metrics_mean(running_metrics, i)

                if ret_last_batch:
                    return running_metrics, (batch, result)
                else:
                    return running_metrics

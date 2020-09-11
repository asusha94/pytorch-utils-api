

class MetricsAverageAbstract:
    def apply(self, running_metrics, metrics, step):
        raise NotImplementedError()

    def __call__(self, running_metrics, metrics):
        self._step += 1
        self.apply(running_metrics, metrics, self._step)

    def _do_init(self):
        self._step = 0


class MetricsEMA(MetricsAverageAbstract):
    '''Exponential moving average'''

    def __init__(self, alpha=0.1):
        self._a = alpha

    def apply(self, running_metrics, metrics, step):
        for k, v in metrics.items():
            if k not in running_metrics:
                running_metrics[k] = (1 - self._a) * v
            else:
                running_metrics[k] = (1 - self._a) * v + self._a * running_metrics[k]

            running_metrics[k] /= (1 - self._a**step)


get_metrics_ema = MetricsEMA


class MetricsCMA(MetricsAverageAbstract):
    '''Cumulative moving average'''

    def apply(self, running_metrics, metrics, step):
        for k, v in metrics.items():
            if k not in running_metrics:
                running_metrics[k] = v / step  # assume that previous are zeros
            else:
                rv = running_metrics[k]
                running_metrics[k] = rv + (v - rv) / step


get_metrics_cma = MetricsCMA


class MetricsSum(MetricsAverageAbstract):
    '''Simple accumulation'''

    def apply(self, running_metrics, metrics, step):
        for k, v in metrics.items():
            if k not in running_metrics:
                running_metrics[k] = v
            else:
                running_metrics[k] = running_metrics[k] + v


get_metrics_sum = MetricsSum


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


def evaluate(model, dataset, *, step_func, calc_metrics,
             metrics_average=None, metrics_map=None,
             ret_last_batch=False, device=None):
    import torch
    from . import model as model_utils

    if not isinstance(step_func, model_utils._ForwardStepWrapper):
        step_func = model_utils._ForwardStepWrapper(step_func)

    if not isinstance(calc_metrics, model_utils._CalcMetricsWrapper):
        calc_metrics = model_utils._CalcMetricsWrapper(calc_metrics)

    if metrics_average is None:
        metrics_average = get_metrics_cma()

    assert isinstance(metrics_average, MetricsAverageAbstract), 'Must derive `MetricsAverageAbstract`'

    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)

    with model_utils.evaluating(model):
        with torch.no_grad():
            i = 1
            running_metrics = dict()

            metrics_average._do_init()
            for i, batch in enumerate(dataset, 1):
                batch = [item.to(device) for item in batch]

                metrics, result = evaluate_batch(model, batch,
                                                 step_func=step_func,
                                                 calc_metrics=calc_metrics,
                                                 ret_result=True)

                metrics_average(running_metrics, metrics)
            else:
                if metrics_map is not None:
                    running_metrics = metrics_map(running_metrics, i)

                if ret_last_batch:
                    return running_metrics, (batch, result)
                else:
                    return running_metrics

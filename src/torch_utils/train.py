

def save_checkpoint(checkpoint_path, *, model, optimizer, step, loss, symlink_name=None):
    import os
    import numpy as np
    import random
    import torch

    rng_state = dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state()
    )

    checkpoint_dict = dict(
        rng_state=rng_state,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),
        step=step,
        loss=loss
    )

    torch.save(checkpoint_dict, checkpoint_path)

    if symlink_name is not None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_name = os.path.basename(checkpoint_path)

        symlink_path = os.path.join(checkpoint_dir, symlink_name)
        if os.path.exists(symlink_path):
            os.unlink(symlink_path)

        os.symlink(checkpoint_name, symlink_path)


def load_checkpoint(checkpoint_path, *, model, optimizer=None, strict=False):
    import os
    import numpy as np
    import random
    import torch

    assert os.path.isfile(checkpoint_path)

    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    rng_state = checkpoint_dict['rng_state']

    random.setstate(rng_state['python'])
    np.random.set_state(rng_state['numpy'])
    torch.set_rng_state(rng_state['torch'])
    torch.cuda.set_rng_state(rng_state['torch_cuda'])

    model.load_state_dict(checkpoint_dict['state_dict'], strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'], strict=strict)

    step = checkpoint_dict['step']
    loss = checkpoint_dict['loss']

    return step, loss


class _TrainStep:
    @property
    def last_result(self):
        return self._last_result

    def __init__(self, step_func):
        self._step_func = step_func

        self._last_result = None

    def __call__(self, *args, **kwargs):
        self._last_result = self._step_func(*args, **kwargs)
        return self._last_result.loss


def _default_summary_write(writer, model, optimizer, metrics, step, batch, result):
    from . import summary as summary_utils

    summary_utils.write_gradients(writer, model, step)

    n_groups = len(optimizer.param_groups)
    for i, group in optimizer.param_groups:
        for k, v in group.items():
            if n_groups == 1:
                label = f'optimizer/{k}'
            else:
                label = f'optimizer/{i}/{k}'

            writer.add_scalar(label, v, global_step=step)

    for k, v in metrics.items():
        try:
            writer.add_scalar(k, v, global_step=step)
        except Exception:
            pass


def train(*, epochs, model, optimizer, step_func,
          train_dataset, val_dataset=None,
          training_dir=None, checkpoint_path=None,
          calc_metrics=None, summary_write=None,
          device=None, params_ops=None):
    import os
    import time
    import torch
    from torch.utils.tensorboard import SummaryWriter

    from . import eval as eval_utils
    from . import model as model_utils

    if not isinstance(step_func, model_utils.ForwardStepWrapper):
        step_func = model_utils.ForwardStepWrapper(step_func)

    if device is None:
        device = next(model.parameters()).device

    if summary_write is None:
        summary_write = _default_summary_write

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.to(device)

    # load checkpoint

    if checkpoint_path is not None:
        _state = load_checkpoint(checkpoint_path, model=model, optimizer=optimizer)
        learning_rate, step, epoch_offset, loss = _state
    else:
        step, epoch_offset, loss = 0, 0, 0

    if training_dir is None:
        training_dir = './training'

    if not os.path.exists(training_dir):
        os.makedirs(training_dir)

    checkpoint_dir = os.path.join(training_dir, 'ckpts')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # creating summary writers

    train_writer = SummaryWriter(log_dir=os.path.join(training_dir, 'summary', 'train'))
    if val_dataset is None:
        valid_writer = None
    else:
        valid_writer = SummaryWriter(log_dir=os.path.join(training_dir, 'summary', 'valid'))

    print('Training started:', type(model).__name__)

    if epoch_offset == 0:
        with model_utils.evaluating(model):
            with torch.no_grad():
                # train metrics
                batch = next(iter(train_dataset))
                batch = [item.to(device) for item in batch]

                metrics, result = eval_utils.evaluate_batch(model, batch,
                                                            step_func=step_func,
                                                            calc_metrics=calc_metrics,
                                                            ret_result=True)
                summary_write(train_writer, model, optimizer, metrics, epoch_offset, batch, result)
                train_writer.flush()

                loss = metrics['loss']

                # valid metrics
                if valid_writer is not None:
                    batch = next(iter(val_dataset))
                    batch = [item.to(device) for item in batch]

                    metrics, result = eval_utils.evaluate_batch(model, batch,
                                                                step_func=step_func,
                                                                calc_metrics=calc_metrics,
                                                                ret_result=True)
                    summary_write(valid_writer, model, optimizer, metrics, epoch_offset, batch, result)
                    valid_writer.flush()

                print(f'epoch #0/{epochs}', f'loss: {loss:.3f}', flush=True)

                # zero checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_0.pth')
                save_checkpoint(checkpoint_path,
                                model=model,
                                optimizer=optimizer,
                                step=0,
                                loss=loss,
                                symlink_name='checkpoint_last.pth')
    else:
        print(f'epoch #{epoch_offset}/{epochs}', f'loss: {loss:.3f}', flush=True)

    train_step = _TrainStep(step_func)

    # train loop
    for epoch in range(epoch_offset, epochs):
        running_metrics = dict()

        epoch_step = 1
        epoch_timer_start = time.time()

        model.train()
        for epoch_step, batch in enumerate(train_dataset, 1):
            step += 1

            batch = [item.to(device) for item in batch]

            def closure():
                # zero the parameter gradients
                optimizer.zero_grad()

                loss = train_step(model, batch)

                loss.backward()

                if params_ops:
                    for op in params_ops:
                        op(model.parameters())

                return loss

            optimizer.step(closure)

            metrics = calc_metrics(model, batch, train_step.last_result)

            eval_utils.metrics_ema(running_metrics, metrics, step=epoch_step)
        else:
            epoch += 1  # due to the end of the epoch

            loss = running_metrics['loss']

            elapsed = time.time() - epoch_timer_start

            # train metrics
            summary_write(train_writer, model, optimizer, running_metrics, epoch, batch, train_step.last_result)
            train_writer.flush()

            print(f'epoch #{epoch}/{epochs}',
                  f'loss: {loss:.3f}', '--',
                  f'elapsed {elapsed:.3f} sec.', flush=True)

            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pth')
            save_checkpoint(checkpoint_path,
                            model=model,
                            optimizer=optimizer,
                            step=epoch,
                            loss=loss,
                            symlink_name='checkpoint_last.pth')

            if valid_writer is not None:
                # valid metrics
                metrics, (batch, result) = eval_utils.evaluate(model, val_dataset,
                                                               step_func=step_func,
                                                               calc_metrics=calc_metrics,
                                                               ret_last_batch=True, device=device)
                summary_write(valid_writer, model, optimizer, metrics, epoch, batch, result)
                valid_writer.flush()
    else:
        print('Training finished')

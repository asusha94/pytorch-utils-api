

def write_gradients(writer, model, step):
    import numpy as np
    import torch

    for name, param in model.named_parameters():
        grad_value = param.grad
        if grad_value is None:
            grad_value = torch.zeros_like(param.data)

        grad_value = grad_value.reshape(-1)
        grad_norm = torch.norm(grad_value, 2).item()

        if not np.isfinite(grad_norm):
            grad_value[grad_value != grad_value] = 0

        writer.add_scalar('gradients-norm/%s' % name, grad_norm, global_step=step)
        writer.add_histogram('gradients-hist/%s' % name, grad_value, global_step=step)


def draw_confusion_matrix(cm, classes=['0', '1'], normalize=True, cmap='Blues'):
    import itertools
    import matplotlib.pyplot as plt
    import numpy as np

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(dpi=96)

    ax = fig.gca()

    ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0., vmax=1.)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks, minor=False)
    _ = [l.set_fontsize(13) for l in ax.set_xticklabels(classes, fontdict=None, minor=False)]
    ax.set_yticks(tick_marks, minor=False)
    _ = [l.set_fontsize(13) for l in ax.set_yticklabels(classes, fontdict=None, minor=False)]

    fmt = '.2f' if normalize else 'd'
    thresh = 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, f'%{fmt}' % cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                size=16)

    ax.set_xlabel('Predicted label').set_fontsize(13)
    ax.set_ylabel('True label').set_fontsize(13)
    fig.tight_layout()

    fig.canvas.draw()

    canvas = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    canvas = canvas.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return canvas

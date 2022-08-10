import os
import os.path as osp
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from configs.config_global import FIG_DIR, ROOT_DIR

plt.rcParams.update({'font.size': 16})
line_styles = ['-', '--', ':']
# Default colors
# colors = ['red', 'tomato', 'green', 'lightgreen', 'blue', 'lightblue']

if not osp.exists(FIG_DIR):
    os.makedirs(FIG_DIR)


def adjust_figure(ax=None):
    if ax is None:
        ax = plt.gca()
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    # ax.yaxis.set_ticks_position('left')
    # ax.xaxis.set_ticks_position('bottom')
    plt.tight_layout(pad=0.5)

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def get_sem(data):
    return np.std(data) / np.sqrt(len(data))

def errorbar_plot(
    x_axis, 
    data_list,
    x_label,
    y_label='',
    label_list=[None, ],
    fig_dir='',
    legend_title=None,
    fig_name=None,
    yticks=None,
    linewidth=2,
    capsize=4,
    capthick=2,
    figsize=(6, 5),
    special_index=-1
):
    
    plt.figure(figsize=figsize)

    for i, data, label in zip(range(len(data_list)), data_list, label_list):

        if i != special_index:
            mean = [np.mean(val) for val in data]
            sem = [get_sem(val) for val in data]

            plt.errorbar(
                x_axis, 
                mean, sem,
                label=label,
                linewidth=linewidth,
                capsize=capsize,
                capthick=capthick
            )
        else:
            
            plt.plot(
                x_axis,
                data,
                label=label,
                linestyle='dotted',
                linewidth=linewidth,
                marker='s',
                color='gray'
            )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(title=legend_title)

    if yticks is not None:
        plt.yticks(yticks)
    
    adjust_figure()

    os.makedirs(osp.join(FIG_DIR, fig_dir), exist_ok=True)
    plt.savefig(osp.join(FIG_DIR, fig_dir, f'{fig_name}.pdf'), transparent=True)
    plt.close()

def error_range_plot(
    x_axis,
    data_list,
    fig_name, 
    fig_dir, 
    x_label,
    y_label,
    plot_title=None, 
    legend=True,
    color_list=['orange', 'blue', 'green', 'tomato'],
    fill_vertical=None,
    label_list=[None, ],
    font=14, 
    linewidth=2.5, 
    x_ticks=None,
    y_ticks=None,
    ylim=None,
    hline=None,
    fig_size=(6, 5)
):
    """
    A general plot for error range
    :param data_list: list of numpy arrays of shpae (n, len)
    """

    fig_dir = osp.join(FIG_DIR, fig_dir)
    plt.figure(figsize=fig_size)

    n = len(data_list)
    for i, (data, label) in enumerate(zip(data_list, label_list)):

        mean = np.mean(data, axis=1)
        sem = np.std(data, axis=1) / np.sqrt(len(data[0]))

        if color_list is not None:
            color = color_list[i]
        else:
            color = None

        plt.plot(x_axis, mean, label=label, linewidth=linewidth, color=color)
        plt.fill_between(
            x=x_axis,
            y1=mean - sem,
            y2=mean + sem,
            alpha=0.25,
            color=color
        )

    if fill_vertical is not None:
        plt.axvspan(-fill_vertical[0] - 0.5, fill_vertical[1] - 0.5, color='black', alpha=0.1)

    plt.xlabel(x_label, fontsize=font)
    plt.ylabel(y_label, fontsize=font)

    if x_ticks is not None:
        plt.xticks(x_ticks, fontsize=font)

    if y_ticks is not None:
        plt.yticks(y_ticks, fontsize=font)

    if hline is not None:
        plt.hlines(hline, min(x_ticks), max(x_ticks), color='black')

    if ylim is not None:
        plt.ylim(*ylim)

    if legend:
        plt.legend()

    if plot_title is not None:
        plt.title(plot_title, fontsize=font)

    adjust_figure()
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(osp.join(fig_dir, f'{fig_name}.pdf'), transparent=True)
    plt.close()

def plot_gif(data, fig_path='', fig_name='', fig_title=''):
    """
    Draw a GIF from torch.Tensor
    """

    fig, ax = plt.subplots()
    def update(i):
        ax.imshow(data[i].cpu().permute(1, 2, 0).numpy())

    ax.set_title(fig_title)
    anim = FuncAnimation(fig, update, np.arange(data.shape[0]), interval=200)

    fig_path = osp.join(FIG_DIR, fig_path)
    os.makedirs(fig_path, exist_ok=True)
    anim.save(osp.join(fig_path, f'{fig_name}.gif'), dpi=80, writer='imagemagick')

def plot_img(img, fig_path='', fig_name='', fig_title=''):
    """
    Draw an image from torch.Tensor
    """

    fig, ax = plt.subplots()

    ax.set_title(fig_title)
    ax.imshow(img.detach().cpu().numpy())

    fig_path = osp.join(FIG_DIR, fig_path)
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(osp.join(fig_path, f'{fig_name}.png'))
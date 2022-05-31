from collections import defaultdict
from ..util import AutoMap
import matplotlib.pyplot as plt

COLORS = defaultdict(lambda: AutoMap(plt.get_cmap('Set1').colors + plt.get_cmap('Set2').colors))
LINES = defaultdict(lambda: AutoMap(['-', '--', ':', '-.']))
MARKERS = defaultdict(lambda: AutoMap(['o', 's', 'v', '^', '<', '>', 'P', '*', 'd', 'h', 'x']))
# LINESTYLES = defaultdict(lambda: AutoMap(product(['-', '--', ':'], ['.', 's', '*'])))
colors_dict = {
    'Random': 'saddlebrown',
    'ActGrad': 'firebrick', #'deeppink',
    'WeightNorm': 'gold',
    'LayerRandom': 'peru',
    'SeqInChange': 'cyan',
    'AsymInChange': 'blue',
    'LayerActGrad': 'red',
    'LayerInChange': 'purple',
    'LayerWeightNorm': 'darkorange',
    'LayerGreedyFS': 'green',
    'LayerGreedyFS-fd': 'lime',
    'LayerSampling': 'grey',
}

markers_dict = {
    'Random': 'P',
    'ActGrad': 'v',
    'WeightNorm': 's',
    'LayerRandom': 'x',
    'SeqInChange': 'h',
    'AsymInChange': 'o',
    'LayerActGrad': '^',
    'LayerInChange': 'd',
    'LayerWeightNorm': 's',
    'LayerGreedyFS': '<',
    'LayerGreedyFS-fd': '>',
    'LayerSampling': '*',
}


def reset_plt():
    global COLORS, LINES, MARKERS
    COLORS = defaultdict(lambda: AutoMap(plt.get_cmap('Set1').colors + plt.get_cmap('Set2').colors))
    LINES = defaultdict(lambda: AutoMap(['-', '--', ':', '-.']))
    MARKERS = defaultdict(lambda: AutoMap(['o', 's', 'v', '^', '<', '>', 'P', '*', 'd', 'h', 'x']))


def plot_df(df,
            x_column,
            y_column,
            colors=None,
            lines=None,
            markers=None,
            delta=False,
            fig=True,
            x_markers=None,
            line=None,
            marker=None,
            color=None,
            suffix='',
            aggregate=True, groupby_col=None, **plt_kwargs):
    global COLORS, LINES, MARKERS, LINESTYLES

    colors_column = colors
    lines_column = lines
    markers_column = markers
    groupby_col = x_column if groupby_col is None else groupby_col
    
    if fig:
        plt.figure(figsize=(15, 8))

    groups = []
    if colors_column is not None:
        groups.append(colors_column)
    if lines_column is not None:
        groups.append(lines_column)
    if markers_column is not None:
        groups.append(markers_column)

    labels = set()

    for items, dfg in df.sort_values(by=groups).groupby(groups):
        if not isinstance(items, tuple):
            items = (items, )

        kwargs = {}
        kwargs.update(plt_kwargs)
        label = ""
        if colors_column is not None:
            i, *items = items
            label += f"{i} - "
            kwargs['color'] = COLORS[colors_column][i] if i not in colors_dict else colors_dict[i]

        if lines_column is not None:
            i, *items = items
            if lines_column != colors_column:
                label += f"{i} - "
            kwargs['ls'] = LINES[lines_column][i]

        if markers_column is not None:
            i, *items = items
            if markers_column != colors_column and markers_column != lines_column:
                label += f"{i} - "
            kwargs['marker'] = MARKERS[markers_column][i] if i not in markers_dict else markers_dict[i]

        if color is not None:
            kwargs['color'] = color

        if line is not None:
            kwargs['ls'] = line

        if marker is not None:
            kwargs['marker'] = marker

        label = label[:-3] + suffix

        dfg = dfg.sort_values(x_column, ascending=True)
        if aggregate:

            if 'label' not in kwargs:
                kwargs['label'] = label

            # dfg = dfg[[x_column, y_column]]
            # mean = dfg.groupby(x_column, as_index=False).mean()
            # std = dfg.groupby(x_column, as_index=False).std()
            mean = dfg.groupby(groupby_col, as_index=False).mean()
            std = dfg.groupby(groupby_col, as_index=False).std()

            x, y = mean[x_column].values, mean[y_column].values
            yerr = std[y_column]
            if delta:
                y = -(y - y[0])
            plt.errorbar(x, y,  yerr=yerr, **kwargs)
        else:
            for _, dfg_ in dfg.groupby('seed'):
                if 'label' not in kwargs:
                    if label not in labels:
                        labels.add(label)
                        kwargs['label'] = label
                    else:
                        kwargs['label'] = None
                dfg_ = dfg_.sort_values(x_column)
                plt.plot(dfg_[x_column].values, dfg_[y_column].values, **kwargs)

    # if y_column in ('pre_acc1', 'post_acc1', 'pre_acc5', 'post_acc5'):
    #     plt.plot(df[x_column].values, (df[df['fraction'] == 1][y_column].iloc[0],) * len(df[x_column].values),
    #              ls='-', color='k')
    #     plt.plot(df[x_column].values,
    #              (df[df['fraction'] == 1][y_column].iloc[0] * 0.95,) * len(df[x_column].values),
    #              ls='--', color='k')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    if x_column in ('compression', 'speedup', 'fraction'):
        plt.xscale('log')
    ticks = sorted(set(df[x_column]))
    plt.xticks(ticks, map(str, ticks))
    # plt.legend()

import json
import pathlib
import string
import pandas as pd
import os

strategy_name = {
'LayerInChangeChannel - 2 - greedy - rwchange': 'LayerInChange',
'LayerInChangeChannel - 2 - greedy - rwchange - sequential': 'SeqInChange',
'LayerInChangeChannel - 2 - asymmetric - greedy - rwchange - sequential': 'AsymInChange',
# 'LayerGreedyFSChannel' : 'LayerGreedyFS',
# 'LayerGreedyFSChannel - 0' : 'LayerGreedyFS',
'LayerGreedyFSChannel - scale_masks' : 'LayerGreedyFS',
'LayerGreedyFSChannel - 0 - scale_masks' : 'LayerGreedyFS',
'LayerGreedyFSChannel - full_data - scale_masks' : 'LayerGreedyFS-fd',
'LayerGreedyFSChannel - full_data - fw - scale_masks': 'LayerGreedyFS-fd',
'LayerGreedyFSChannel - fw - scale_masks': 'LayerGreedyFS',
'LayerSamplingChannel - 1e-12': 'LayerSampling',
'LayerSamplingChannel - 1e-16': 'LayerSampling',
'LayerActGradChannel':  'LayerActGrad',
'ActGradChannel - norm': 'ActGrad',
'LayerWeightNormChannel - 1' : 'LayerWeightNorm',
'WeightNormChannel - 1 - norm': 'WeightNorm',
'LayerRandomChannel': 'LayerRandom',
'RandomChannel': 'Random',
}

# def strategy_name(strategy, prune_kwargs):
#     if strategy == 'LayerInChangeChannel':
#         return 'LayerInChange' if prune_kwargs['sequential'] is False else \
#             ('SeqInChange' if prune_kwargs['asymmetric'] is False else 'AsymInChange') + prune_kwargs.get('epsilon', 0)


def param_label(key, value):
    if isinstance(value, bool):
        return f' - {key}' if value else ''
    elif key in ["onelayer_results_dir", "select", "patches", "train_dl"]:
        return ''
    else:
        return f' - {value}'


def df_from_results(results_path, glob='*', cf_key='compression', structured=True, icml=False):
    results = []
    results_path = pathlib.Path(results_path)

    COLUMNS = ['dataset', 'model', 'prune_layers',
               'strategy', cf_key, 'pruning_time',
               'size', 'size_nz_orig', 'size_nz', 'real_compression',
               'flops', 'flops_nz', 'speedup',
               'pre_acc1', 'pre_acc5', 'post_acc1', 'post_acc5', 'last_acc1', 'last_acc5', 'finetuning_time',
               # 'train_acc1_all', 'train_acc5_all', 'train_loss_all',
               # 'val_acc1_all', 'val_acc5_all', 'val_loss_all',
               'seed', 'nbatches', 'batch_size', 'train_kwargs',  # 'prune_kwargs',
               'completed_epochs', 'path']
    # Structured pruning experiments have fraction (channels kept/ prunable channels) param instead of the compression
    # (total weights / pruned weights) param used in weight pruning experiments

    if structured:
        COLUMNS += ['reweight', 'structure']

    count_skipped = 0
    for exp in results_path.glob(glob):
        if exp.is_dir():
            try:
                if os.path.isfile(exp / 'params.json') and os.path.isfile(exp / 'metrics.json') and os.path.isfile(exp / 'logs.csv'):
                    with open(exp / 'params.json', 'r') as f:
                        params = eval(json.load(f)['params'])
                    with open(exp / 'metrics.json', 'r') as f:
                        metrics = json.load(f)
                    logs = pd.read_csv(exp / 'logs.csv')

                    strategy = params['strategy'] + ''.join(sorted([param_label(k, v) for k, v in params['prune_kwargs'].items()]))
                    row = [
                        # Params
                        params['dataset'],
                        params['model'],
                        ', '.join(params['prune_layers']),
                        strategy_name.get(strategy, 'NotIncluded') if icml else strategy,
                        params[cf_key],
                        # Metrics
                        metrics['pruning_time'],
                        metrics['size'],
                        metrics['size_nz_orig'] if 'size_nz_orig' in metrics.keys() else None,
                        metrics['size_nz'],
                        metrics['compression_ratio'],
                        metrics['flops'],
                        metrics['flops_nz'],
                        metrics['theoretical_speedup'],
                        # Pre Accs
                        metrics['val_acc1'],
                        metrics['val_acc5'],
                        # Post Accs
                        logs['val_acc1'].max(),
                        logs['val_acc5'].max(),
                        # Last Post Accs
                        logs['val_acc1'].iloc[-1],
                        logs['val_acc5'].iloc[-1],
                        # fine tuning time
                        logs['timestamp'].iloc[-1] if len(logs) > 1 else 0,
                        # # All post Accs and loss
                        # logs['train_acc1'].iloc[:],
                        # logs['train_acc5'].iloc[:],
                        # logs['train_loss'].iloc[:],
                        # logs['val_acc1'].iloc[:],
                        # logs['val_acc5'].iloc[:],
                        # logs['val_loss'].iloc[:],
                        # Other params
                        params['seed'],
                        params['nbatches'],
                        params['dl_kwargs']['batch_size'],
                        # params['train_kwargs']['epochs'],
                        # params['train_kwargs']['optim'],
                        params['train_kwargs'],
                        # ''.join([param_label(k, v) for k, v in params['prune_kwargs'].items()]),
                        len(logs), #Completed epochs
                        str(exp),
                    ]
                    if structured:
                        row += [params['reweight'], params['structure']]
                    results.append(row)
                else:
                    print(f"{exp} is missing one of these files: params, metrics, logs!")
                    count_skipped+=1
            except Exception as inst:
                print(type(inst))
                count_skipped += 1

    print("number of folders skipped: ", count_skipped)
    df = pd.DataFrame(data=results, columns=COLUMNS)
    # df = broadcast_unitary_compression(df, compression_key)
    df = df.sort_values(by=['dataset', 'model', 'strategy', cf_key, 'seed'])
    return df


def df_filter(df, **kwargs):

    for k, vs in kwargs.items():
        if not isinstance(vs, list):
            vs = [vs]
        df = df[getattr(df, k).isin(vs)]
        # for v in vs:
        #     selector |= (df == v)
        # df = df[selector]
    return df


def broadcast_unitary_compression(df, cf_key):
    for _, row in df[df[cf_key] == 1].iterrows():
        for strategy in set(df['strategy']):
            if strategy is not None:
                new_row = row.copy()
                new_row['strategy'] = strategy
                df = df.append(new_row, ignore_index=True)
    return df

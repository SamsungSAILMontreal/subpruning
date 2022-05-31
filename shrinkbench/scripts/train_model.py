import os
import torch
from shrinkbench.experiment import PruningExperiment, StructuredPruningExperiment
from shrinkbench.plot import df_from_results, plot_df
from torchvision import transforms
import pandas as pd
import argparse
import pathlib

parser = argparse.ArgumentParser(description='Train NN')
parser.add_argument('--job', dest='job_id', type=int, default=1)
parser.add_argument('--path', dest='path', type=str, help='path to save/read results', default='results')
parser.add_argument('--data_path', dest='data_path', type=str, help='path to dataset', default='../data')

# train models without pruning
# checkpoints have both model_state_dict and optim_state_dict, save copy with only model_state_dict to load it in models
if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['DATAPATH'] = args.data_path
    os.environ['WEIGHTSPATH'] = '../pretrained/shrinkbench-models'
    os.environ['TORCH_HOME'] = '../pretrained/torchvision-models'

    print(torch.cuda.is_available())

    # model = 'LeNet'
    dataset = 'MNIST'
    model = 'vgg11_bn_small'
    # dataset = 'CIFAR10'
    structure = 'neuron'
    fractions = [1]
    strategies = [('RandomChannel',  {})]

    rootdir = f'{args.path}/{dataset}-{model}-{args.job_id}'
    path = pathlib.Path(rootdir)
    path.mkdir(parents=True, exist_ok=True)

    for strategy, prune_kwargs in strategies:
        for reweight in [False]:
            exp = StructuredPruningExperiment(dataset=dataset,
                                          model=model,
                                          strategy=strategy,
                                          fractions=fractions,
                                          reweight=reweight,
                                          bias=False,
                                          structure=structure,
                                          prune_layers=[],
                                          nbatches=4,
                                          prune_kwargs=prune_kwargs,
                                          train_kwargs={'epochs': 0} if dataset == 'MNIST' and model == 'LeNet' else
                                                       {'epochs': 200, 'optim': 'Adam', 'scheduler': 'MultiStepLR',
                                                        'optim_kwargs': {'lr': 1e-3, 'weight_decay': 5e-4},
                                                        'scheduler_kwargs': {'gamma': 0.1, 'milestones': [100, 150]}},
                                          dl_kwargs={'cifar_shape': True},
                                          rootdir=rootdir,
                                          pretrained=True,  # False
                                          finetune=True,
                                          seed=42)

            exp.run()
    structured = True
    cf_key = 'fraction' if structured else 'compression'
    df = df_from_results(rootdir, structured=structured)
    df = df[(df['model'] == model) & (df['dataset'] == dataset)]

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

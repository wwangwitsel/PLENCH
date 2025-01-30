import argparse
import collections
import json
import os
import random
import sys
import time
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from .data import datasets
from .core import hparams_registry, algorithms
from .lib import misc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Partial-label learning')
    parser.add_argument('--data_dir', type=str, default="")
    parser.add_argument('--dataset', type=str, default="PLCIFAR10_Aggregate")
    parser.add_argument('--algorithm', type=str, default="PRODEN")
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=60000,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=1000,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.1)
    parser.add_argument('--tabular_test_fraction', type=float, default=0.1, required=False)
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--pl_type', help='partial label generation type', default='real', type=str, choices=['uss','fps', 'real'], required=False)
    parser.add_argument('--fps_rate', help='partial label generation flip rate of fps', default=0, type=float, required=False)
    args = parser.parse_args()
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    hparams['max_steps']=args.steps
    hparams['output_dir']=args.output_dir
    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)



    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in datasets.IMAGE_DATASETS and args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir, args)
        test_dataset = datasets.test_dataset_gen(args.data_dir, args)      
    elif args.dataset in datasets.TABULAR_DATASETS:
        dataset, test_dataset = datasets.tabular_train_test_dataset_gen(root=args.data_dir, seed=misc.seed_hash(args.trial_seed), args=args)
    else:
        raise NotImplementedError

    val_dataset, train_dataset = misc.split_dataset(dataset, int(len(dataset)*args.holdout_fraction), misc.seed_hash(args.trial_seed))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=hparams['batch_size'], shuffle=True,num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=64, shuffle=True,num_workers=0, drop_last=False)

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(train_dataset.data.shape, train_dataset.partial_targets, hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)
    train_minibatches_iterator = iter(train_loader)
    checkpoint_vals = collections.defaultdict(lambda: [])
    steps_per_epoch = len(train_dataset)/hparams['batch_size']
    n_steps = args.steps 
    checkpoint_freq = args.checkpoint_freq

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": train_dataset.data.shape,
            "model_num_classes": dataset.num_classes,
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    last_results_keys = None
    for step in range(start_step, n_steps):
        algorithm.train()
        step_start_time = time.time()
        if args.algorithm == 'ALIM':
            try:
                minibatches_device1 = [item.to(device) for item in next(train_minibatches_iterator)]
            except:
                train_minibatches_iterator = iter(train_loader)
                minibatches_device1 = [item.to(device) for item in next(train_minibatches_iterator)]
            try:
                minibatches_device2 = [item.to(device) for item in next(train_minibatches_iterator)]
            except:
                train_minibatches_iterator = iter(train_loader)
                minibatches_device2 = [item.to(device) for item in next(train_minibatches_iterator)]
            step_vals = algorithm.update(minibatches_device1, minibatches_device2)            
        else:
            try:
                minibatches_device = [item.to(device) for item in next(train_minibatches_iterator)]
            except:
                train_minibatches_iterator = iter(train_loader)
                minibatches_device = [item.to(device) for item in next(train_minibatches_iterator)]
            step_vals = algorithm.update(minibatches_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            val_acc = misc.val_accuracy(algorithm, val_loader, device) 
            results['val_accuracy'] = val_acc
            val_covering_rate = misc.val_covering_rate(algorithm, val_loader, device)
            results['val_covering_rate'] = val_covering_rate
            val_approximated_acc = misc.val_approximated_accuracy(algorithm, val_loader, device)
            results['val_approximated_acc'] = val_approximated_acc   
            acc = misc.accuracy(algorithm, test_loader, device)
            results['test_acc'] = acc
            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys], colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, cls=misc.NpEncoder, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

    save_checkpoint('model.pkl')
    # delete VALEN adj file
    if args.algorithm == 'VALEN':
        adj_file_path = os.path.join(args.output_dir, 'adj_matrix.npy')
        if os.path.exists(adj_file_path):
            os.remove(adj_file_path)

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
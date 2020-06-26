import argparse
import logging
import os
from typing import List, Dict, Any

# Hide TF1 future warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import tensorflow as tf
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)

import numpy as np

from graphdg import tools, tf_tools, gcvae_tools
from graphdg.embed import embed_conformer
from graphdg.gcvae import GraphCVAE
from graphdg.parse.distances import get_dataset
from graphdg.standardize import GraphStandardizer, ArrayStandardizer
from graphdg.tf_tools import SessionIO
from graphdg.tools import select_unique_mols, compute_stats, DictSaver, load_pickle, split_by_edges, \
    write_conformers_to_xyz, align_conformers
from graphdg.train import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Command line tool of GraphDG')

    # Name and seed
    parser.add_argument('--name', help='experiment name', required=False, default='graphdg')
    parser.add_argument('--seed', help='run ID', type=int, default=0)

    # Directories
    parser.add_argument('--log_dir', help='directory for log files', type=str, default='logs')
    parser.add_argument('--model_dir', help='directory for model files', type=str, default='models')
    parser.add_argument('--results_dir', help='directory for results', type=str, default='results')
    parser.add_argument('--xyz_dir', help='directory for XYZ files', type=str, default='xyz')

    # Dataset
    parser.add_argument('--train_path', help='path to training dataset', type=str, required=True)
    parser.add_argument('--test_path', help='path to test dataset', type=str, required=True)
    parser.add_argument('--valid_fraction',
                        help='proportion of validation set (0<vs<1)',
                        type=float,
                        required=False,
                        default=0.20)
    parser.add_argument('--num_samples', help='number of samples', type=int, required=False, default=100)
    parser.add_argument('--augmentation',
                        help='number of graphs per molecule in training set',
                        type=int,
                        required=False,
                        default=1)

    # Model
    parser.add_argument('--hidden_feats_edges',
                        help='number of hidden edge features',
                        type=int,
                        required=False,
                        default=10)
    parser.add_argument('--hidden_feats_nodes',
                        help='number of hidden node features',
                        type=int,
                        required=False,
                        default=10)
    parser.add_argument('--latent_dims', help='number of latent dimensions', type=int, required=False, default=1)

    # Encoder
    parser.add_argument('--node_encoder_width',
                        help='width of node encoder model',
                        type=int,
                        required=False,
                        default=50)
    parser.add_argument('--node_encoder_depth', help='depth of node encoder model', type=int, required=False, default=2)
    parser.add_argument('--edge_encoder_width',
                        help='width of edge encoder model',
                        type=int,
                        required=False,
                        default=50)
    parser.add_argument('--edge_encoder_depth', help='depth of edge encoder model', type=int, required=False, default=2)

    # Core
    parser.add_argument('--node_core_width', help='width of node core model', type=int, required=False, default=50)
    parser.add_argument('--node_core_depth', help='depth of node core model', type=int, required=False, default=2)
    parser.add_argument('--edge_core_width', help='width of edge core model', type=int, required=False, default=50)
    parser.add_argument('--edge_core_depth', help='depth of edge core model', type=int, required=False, default=2)
    parser.add_argument('--num_core_models', help='number of core models', type=int, required=False, default=3)

    # Decoder
    parser.add_argument('--decoder_width', help='width of decoder model', type=int, required=False, default=50)
    parser.add_argument('--decoder_depth', help='depth of decoder model', type=int, required=False, default=2)

    parser.add_argument('--batch_size', help='batch size', type=int, required=False, default=32)
    parser.add_argument('--max_num_epochs', help='maximum number of epochs', type=int, required=False, default=100)
    parser.add_argument('--loss_threshold',
                        help='number of times evaluation loss may increase',
                        type=int,
                        required=False,
                        default=10)

    # IO
    parser.add_argument('--load_model', help='load model from checkpoint file', action='store_true', default=False)

    # Logging
    parser.add_argument('--log_level', help='log level', type=str, default='INFO')

    return parser.parse_args()


def get_config() -> dict:
    return vars(parse_args())


def get_test_loss(model: GraphCVAE, session: tf.Session, test_targets: List, test_dicts: List,
                  batch_size: int) -> float:
    test_loss = 0
    for batch in tools.get_batches(test_targets, batch_size=batch_size, shuffle=False):
        test_fd = model.get_train_feed_dict((test_targets, test_dicts), indices=batch, is_training=False)
        test_loss += session.run(model.loss, feed_dict=test_fd)

    return test_loss


def main() -> None:
    config = get_config()

    # Setup
    tools.create_directories([config['log_dir'], config['model_dir'], config['results_dir'], config['xyz_dir']])
    tf_tools.set_seeds(config['seed'])

    tag = tools.get_tag(config)
    tools.setup_logger(log_level=config['log_level'], file_path=os.path.join(config['log_dir'], tag + '.log'))
    tools.save_config(config, file_path=os.path.join(config['log_dir'], tag + '.json'))

    # Load train and test sets
    logging.info('Loading training and test datasets')
    train_mols = load_pickle(path=config['train_path'])
    test_mols = load_pickle(path=config['test_path'])

    # Remove duplicates
    test_mols = select_unique_mols(test_mols)

    # Generate datasets
    train_dataset = get_dataset(train_mols, seed=config['seed'], count=config['augmentation'])
    test_dataset = get_dataset(test_mols, seed=config['seed'], count=1)
    logging.info(f'Dataset sizes: {len(train_dataset)} train, {len(test_dataset)} test')

    # Split into dicts and targets
    train_dicts, train_targets = [d for d, _ in train_dataset], [t for _, t in train_dataset]
    test_dicts, test_targets = [d for d, _ in test_dataset], [t for _, t in test_dataset]

    # Normalize
    graph_standardizer = GraphStandardizer.from_data_dicts(train_dicts)
    array_standardizer = ArrayStandardizer.from_array(np.vstack(train_targets))

    train_dicts_standard = [graph_standardizer.standardize_data_dict(d) for d in train_dicts]
    train_targets_standard = [array_standardizer.standardize(t) for t in train_targets]

    test_dicts_standard = [graph_standardizer.standardize_data_dict(d) for d in test_dicts]
    test_targets_standard = [array_standardizer.standardize(t) for t in test_targets]

    # Build model
    graph_dims = tf_tools.GraphFeatureDimensions.from_data_dicts(train_dicts)
    logging.info(f'Graph input dimensions: {graph_dims}')
    model = gcvae_tools.get_model(graph_dims=graph_dims, config=config)

    session_handler = SessionIO(path=os.path.join(config['model_dir'], tag))

    saver = DictSaver(directory=config['results_dir'], tag=tag)

    with tf.Session(config=tf_tools.get_session_config()) as session:
        trainer = Trainer(loss_fn=model.loss, feed_dict_fn=model.get_train_feed_dict)
        session.run(tf.global_variables_initializer())

        if config['load_model']:
            session_handler.load(session)

        # Train model
        train_info = trainer.train(data=(train_targets_standard, train_dicts_standard),
                                   session=session,
                                   session_handler=session_handler,
                                   batch_size=config['batch_size'],
                                   max_num_epochs=config['max_num_epochs'],
                                   loss_threshold=config['loss_threshold'],
                                   seed=config['seed'],
                                   valid_fraction=config['valid_fraction'])

        logging.info(f'Training statistics: {train_info}')
        saver.save(train_info, name='train')

        # Test model
        test_info: Dict[str, Any] = {
            'test_loss':
            get_test_loss(model=model,
                          session=session,
                          test_targets=test_targets_standard,
                          test_dicts=test_dicts_standard,
                          batch_size=config['batch_size']),
        }

        means_standard, log_vars_standard = model.sample(session,
                                                         data_dicts=test_dicts_standard,
                                                         seed=config['seed'],
                                                         batch_size=config['batch_size'])
        means = array_standardizer.destandardize(means_standard)
        deltas = (means - np.vstack(test_targets))
        test_info['error'] = compute_stats(deltas)

        variances = array_standardizer.std**2 * np.exp(log_vars_standard)
        test_info['variance'] = compute_stats(variances)

        logging.info(f'Test statistics: {test_info}')
        saver.save(test_info, name='test')

        # Remove initial conformers
        for molecule in test_mols:
            molecule.RemoveAllConformers()

        # Generate conformers
        logging.info(f"Attempting to generate {config['num_samples']} sample(s) for each molecule in test set")
        for sample_id in range(config['num_samples']):
            means_standard, log_vars_standard = model.sample(session,
                                                             test_dicts_standard,
                                                             seed=config['seed'] + sample_id,
                                                             batch_size=config['batch_size'])
            means = array_standardizer.destandardize(means_standard)
            variances = array_standardizer.std**2 * np.exp(log_vars_standard)

            means_list = split_by_edges(test_dicts, means)
            variance_list = split_by_edges(test_dicts, variances)

            for molecule, test_dict, means, variances in zip(test_mols, test_dicts, means_list, variance_list):
                embed_conformer(
                    molecule,
                    senders=test_dict[tools.SENDERS],
                    receivers=test_dict[tools.RECEIVERS],
                    means=means,
                    variances=variances,
                    seed=config['seed'] + sample_id,
                )

        # Report statistics on sampling
        num_conformers = [molecule.GetNumConformers() for molecule in test_mols]
        sample_info = compute_stats(np.array(num_conformers))
        logging.info(f'Sample statistics: {sample_info}')
        saver.save(sample_info, name='sample')

        # Write samples to file
        logging.info(f'Writing samples to XYZ')
        for mol_id, molecule in enumerate(test_mols):
            align_conformers(molecule)
            path = os.path.join(config['xyz_dir'], tag + '_' + str(mol_id) + '.xyz')
            write_conformers_to_xyz(molecule, path=path)


if __name__ == '__main__':
    main()

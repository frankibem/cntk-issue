import os
import cntk
import argparse
from resnet import resnet_model
from cntk import learning_parameter_schedule, momentum_schedule
from cntk.io import StreamDef, StreamDefs, MinibatchSource, CBFDeserializer
from cntk.layers import Dense, Sequential, Label, LSTM, Recurrence, BatchNormalization
from cntk.train import Trainer, TestConfig, CrossValidationConfig, training_session, CheckpointConfig, DataUnit

# Model dimensions
frame_height = 80
frame_width = 80
num_channels = 1
num_classes = 84
hidden_dim = 42
model_name = 'boston50.model'

# Dataset partition sizes
test_size = 84
train_size = 399


def cbf_reader(path, is_training, max_samples):
    """
    Returns a MinibatchSource for data at the given path
    :param path: Path to a CBF file
    :param is_training: Set to true if reader is for training set, else false
    :param max_samples: Max no. of samples to read
    """
    deserializer = CBFDeserializer(path, StreamDefs(
        label=StreamDef(field='label', shape=num_classes, is_sparse=True),
        front=StreamDef(field='pixels', shape=num_channels * frame_height * frame_width, is_sparse=False),
    ))

    return MinibatchSource(deserializer, randomize=is_training, max_samples=max_samples)


def bidirectional_recurrence(fwd, bwd):
    f = Recurrence(fwd)
    g = Recurrence(bwd, go_backwards=True)
    x = cntk.placeholder()
    return cntk.splice(f(x), g(x))


def create_network():
    input_var = cntk.sequence.input_variable((num_channels, frame_height, frame_width), name='input_var')
    target_var = cntk.input_variable((num_classes,), is_sparse=True, name='target_var')

    with cntk.layers.default_options(enable_self_stabilization=True):
        model = Sequential([
            resnet_model(cntk.placeholder()), Label('resnet'),
            Dense(hidden_dim, name='cnn_fc'),
            cntk.layers.Stabilizer(),
            bidirectional_recurrence(LSTM(hidden_dim // 2), LSTM(hidden_dim // 2)),
            cntk.sequence.last,
            BatchNormalization(),
            Dense(num_classes)
        ])(input_var)

    return {
        'input': input_var,
        'target': target_var,
        'model': model,
        'loss': cntk.cross_entropy_with_softmax(model, target_var),
        'metric': cntk.classification_error(model, target_var)
    }


def main(params):
    # Create output and log directories if they don't exist
    if not os.path.isdir(params['output_folder']):
        os.makedirs(params['output_folder'])

    if not os.path.isdir(params['log_folder']):
        os.makedirs(params['log_folder'])

    # Create the network
    network = create_network()

    # Create readers
    train_reader = cbf_reader(os.path.join(params['input_folder'], 'train{}.cbf'.format(params['prefix'])), is_training=True,
                              max_samples=cntk.io.INFINITELY_REPEAT)
    cv_reader = cbf_reader(os.path.join(params['input_folder'], 'test{}.cbf'.format(params['prefix'])), is_training=False,
                           max_samples=cntk.io.FULL_DATA_SWEEP)
    test_reader = cbf_reader(os.path.join(params['input_folder'], 'test{}.cbf'.format(params['prefix'])), is_training=False,
                             max_samples=cntk.io.FULL_DATA_SWEEP)

    input_map = {
        network['input']: train_reader.streams.front,
        network['target']: train_reader.streams.label
    }

    # Create learner
    mm_schedule = momentum_schedule(0.90)
    lr_schedule = learning_parameter_schedule([(40, 0.1), (40, 0.01)], minibatch_size=params['minibatch_size'])
    learner = cntk.adam(network['model'].parameters, lr_schedule, mm_schedule, l2_regularization_weight=0.0005,
                        epoch_size=params['epoch_size'], minibatch_size=params['minibatch_size'])

    # Use TensorBoard for visual logging
    log_file = os.path.join(params['log_folder'], 'log.txt')
    pp_writer = cntk.logging.ProgressPrinter(freq=10, tag='Training', num_epochs=params['max_epochs'], log_to_file=log_file)
    tb_writer = cntk.logging.TensorBoardProgressWriter(freq=10, log_dir=params['log_folder'], model=network['model'])

    # Create trainer and training session
    trainer = Trainer(network['model'], (network['loss'], network['metric']), [learner], [pp_writer, tb_writer])
    test_config = TestConfig(minibatch_source=test_reader, minibatch_size=params['minibatch_size'], model_inputs_to_streams=input_map)
    cv_config = CrossValidationConfig(minibatch_source=cv_reader, frequency=(1, DataUnit.sweep),
                                      minibatch_size=params['minibatch_size'], model_inputs_to_streams=input_map)
    checkpoint_config = CheckpointConfig(os.path.join(params['output_folder'], model_name), frequency=(10, DataUnit.sweep), restore=params['restore'])

    session = training_session(trainer=trainer,
                               mb_source=train_reader,
                               mb_size=params['minibatch_size'],
                               model_inputs_to_streams=input_map,
                               max_samples=params['epoch_size'] * params['max_epochs'],
                               progress_frequency=(1, DataUnit.sweep),
                               checkpoint_config=checkpoint_config,
                               cv_config=cv_config,
                               test_config=test_config)

    cntk.logging.log_number_of_parameters(network['model'])
    session.train()

    # Save the trained model
    path = os.path.join(params['output_folder'], 'final_model.dnn')
    network['model'].save(path)
    print('Saved final model to', path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-if', '--input_folder', help='Directory where dataset is located', required=False, default='dataset')
    parser.add_argument('-of', '--output_folder', help='Directory for models and checkpoints', required=False, default='models')
    parser.add_argument('-lf', '--log_folder', help='Directory for log files', required=False, default='logs')
    parser.add_argument('-n', '--num_epochs', help='Total number of epochs to train', type=int, required=False, default=80)
    parser.add_argument('-m', '--minibatch_size', help='Minibatch size in samples', type=int, required=False, default=60)
    parser.add_argument('-e', '--epoch_size', help='Epoch size', type=int, required=False, default=train_size)
    parser.add_argument('-r', '--restore', help='Indicates whether to resume from previous checkpoint', action='store_true')
    parser.add_argument('-p', '--prefix', help='The prefix for the train/test datasets', required=False, default='')

    args = parser.parse_args()
    main({
        'input_folder': args.input_folder,
        'output_folder': args.output_folder,
        'log_folder': args.log_folder,
        'max_epochs': args.num_epochs,
        'minibatch_size': args.minibatch_size,
        'epoch_size': args.epoch_size,
        'restore': args.restore,
        'prefix': args.prefix
    })

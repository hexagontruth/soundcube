# Soundcube

Soundcube is a Python project for facilitating the algorithmic generation of musical waveforms from seed sequences, using a recurrent neural network.

# Dependencies

Required:

- [Theano](http://deeplearning.net/software/theano/) or [TensorFlow](https://tensorflow.com/) (not tested with TensorFlow)

- [Keras](http://keras.io/)

- [NumPy](http://www.numpy.org/)

- [SciPy](https://www.scipy.org/)

- [FFmpeg](https://ffmpeg.org/) or other audio conversion tool

Recommended:

- [CUDA](https://developer.nvidia.com/cuda-downloads)

## Quick Start

To train and generate with a model using the default configuration:

  1. Place existing music files in the `music` directory.

  2. Run the following commands, in order, from the project root directory:

          python convert.py
          python train.py
          python generate.py

## Configuration

General configuration parameters can be set in `config.yml`. Note that certain parameters are co-dependent on each other, and not all --- nay, even most --- combinations of possible parameters have been tested. More granular settings vis-a-vis the configuration of NN models, etc., can be set in the appropriate source files.

Most `config.yml` key-value pairs can be set when executing one of the CLI scripts, by passing an argument of the form `key.subkey=value`. For instance, to set the data directory, number of training epochs, and weight file when running `train.py`, one could use:

    python train.py data.target_dir=my_dir \
      net.training_epochs=42 \
      state.model_base=fancy_model

(Note that training epochs can be specified by a standalone integer argument as well.)

### Training

1. Move an appropriate corpus of music to the `my music` directory. I recommend something in the range of 2-4 hours, less certainly if your hardware sucks. Music can be left in separate directories --- the conversion script will flatten all directory hierarchies when converting into the _data_ directory.

2. Once you have satisfied all dependencies and customized the configuration file to your liking (though it should work out of the box), run `python convert.py`.

3. Run `python train.py [epochs]`.  This will presumably take a while, but Keras should keep you appraised of progress through the on-screen progress bar, epoch count, etc.

    - `[epochs]` to train for. If not specified, the value in `config.yml` will be used.

If the default settings are used, training history and weights will be saved periodically, and upon the completion of training. If and when the training script is run again it will resume where it left off, once it rebuilds the model and loads the weight. (Note that model-building can take quite a while each time one of these scripts is executed.) One can train multiple models concurrently either by passing specific model names to the training and generation scripts, or by changing the config file between iterations.

### Generation

1. Run `python generate.py [OUTPUT_FILE_1]...`.

    - Specify one or more output filename.
    - If no filenames are specified, a single output is generated to the default filename.

### Reset

Running `reset.py` will, by default, move the contents of the models directory to _.bak_ files, allowing a new model to be trained to the default filenames. It accepts the following flags:

  - `clear` --- Deletes all model files rather than renaming them (obviously use this with caution).
  - `logs` --- Deletes log file(s).
  - `data` --- Deletes formatted data in `data` (target_dir) directory. Does not affect original music in `music` directory.

### Model Building

Additional models can be defined as model functions in `lib/models.py`. See pydoc documentation for details.

## Interactive Use

All functionality should work in an interactive session. For instance, the entire functionality of the three training and generation scripts can be recapitulated as follows:

    from lib.net import net
    import lib.utils as utils
    utils.convert_source_dir()
    utils.convert_wavs_to_freq()
    n = Net(Net.ALL)
    n.train()
    y = n.gen()
    utils.write_output('output/my_fancy_song.wav', y)

A variety of more fine-grained options are available in both the `utils` module and `Net` class. Consult the documentation associated with these respective files for more details.

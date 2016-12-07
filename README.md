# Soundcube

Soundcube is a Python project for facilitating the algorithmic generation of musical waveforms from seed sequences, using a recurrent neural network.

# Dependencies

Required:

- [Theano](http://deeplearning.net/software/theano/) or [TensorFlow](https://tensorflow.com/) (not tested with TensorFlow)

- [Keras](http://keras.io/)

- [NumPy](http://www.numpy.org/)

- [SciPy](https://www.scipy.org/)

- [FFmpeg](https://ffmpeg.org/) or other audio conversion tool


## Quick Start

Place music in the `music` directory and run the following commands, in order, from the project root directory:

    python convert.py
    python train.py
    python generate.py

## Configuration

General configuration parameters can be set in _config.yml_. Note that certain parameters are co-dependent on each other, and not all --- nay, even most --- combinations of possible parameters have been tested. More granular settings vis-a-vis the configuration of NN models, etc., can be set in the appropriate source files.

## Training

1. Move an appropriate corpus of music to the _my music_ directory. I recommend something in the range of 2-4 hours, less certainly if your hardware sucks. Music can be left in separate directories --- the conversion script will flatten all directory hierarchies when converting into the _data_ directory.

2. Once you have satisfied all dependencies and customized the configuration file to your liking (though it should work out of the box), run `python convert.py`.

3. Run `python train.py [epochs]`.  This will presumably take a while, but Keras should keep you appraised of progress through the on-screen progress bar, epoch count, etc.

    - `[epochs]` to train for. If not specified, the value in _config.yml_ will be used.

## Generation

1. Run `python generate.py [output_file_1|...]`.

    - Specify one or more output filename.
    - If no filenames are specified, a single output is generated to the default filename.
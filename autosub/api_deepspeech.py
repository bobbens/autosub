#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines DeepSpeech API used by autosub
"""
# Import built-in modules
import sys
import subprocess
from pathlib import Path

# Import third-party modules
import shlex
from deepspeech import Model, version
import numpy as np

# Any changes to the path and your own modules
from autosub import exceptions
from autosub import constants


def convert_samplerate(audio_path, desired_sample_rate):
    try:
        from shhlex import quote
    except ImportError:
        from pipes import quote
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_path), desired_sample_rate)
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use {}hz files or install it: {}'.format(desired_sample_rate, e.strerror))

    return desired_sample_rate, np.frombuffer(output, np.int16)


class DeepSpeech:
    def __init__(self,
                 config=None):
        self.config = config
        basepath   = str(Path.home())+'/.local/share/deepspeech/'
        modelpath  = basepath+'latest-models.pbmm'
        scorerpath = basepath+'latest-models.scorer'
        self.ds = Model(modelpath)
        self.desired_sample_rate = self.ds.sampleRate()
        if scorerpath:
            self.ds.enableExternalScorer(scorerpath)

    def __call__(self, filename):
        try:  # pylint: disable=too-many-nested-blocks
            fs_new, audio = convert_samplerate(filename, self.desired_sample_rate)
            return self.ds.stt(audio)
        except KeyboardInterrupt:
            return None
        return None 


# Based on Timbre-Trap & SS-MPE # Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

from . import CQT
from . import HCQT
from . import Onset_Finder
from . import CFP_HMLC

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

import matplotlib.pyplot as plt

class SS_NT(nn.Module):
    """
    self-supervised pitch & onset estimation (note tracking)
    """

    def __init__(self, cqt_params, CFP_HMLC_params, onset_bins_per_semitone, CFP_mode):
        """
        Initialize the full autoencoder.

        Parameters
        ----------
        hcqt_params : dict
          Parameters for feature extraction module
        """

        nn.Module.__init__(self)

        self.bins_per_semitones_OPS = onset_bins_per_semitone

        cqt_params_OPS = cqt_params.copy()
        cqt_params_OPS['bins_per_octave'] = 12 * self.bins_per_semitones_OPS
        cqt_params_OPS['n_bins'] = cqt_params['n_bins'] // (cqt_params['bins_per_octave'] // 12) * self.bins_per_semitones_OPS

        # Copy of module config        
        self.cqt_params = cqt_params.copy()
        self.cqt_params_OPS = cqt_params_OPS.copy()
        self.CFP_HMLC_params = CFP_HMLC_params.copy()

        # STFT/CQT/HCQT for pitch
        cqt_params.pop('weights')
        self.sliCQ = HCQT(**cqt_params)

        # STFT/CQT/HCQT for onset (One bin Per Semitone)
        cqt_params_OPS.pop('weights')
        self.sliCQ_OPS = HCQT(**cqt_params_OPS)

        # onset selection
        self.onset_finder = Onset_Finder(**cqt_params_OPS)

        # CFP_HMLC
        self.CFP_HMLC = CFP_HMLC(**CFP_HMLC_params)

        # modules
        self.encoder = None
        self.decoder = None
        # self.pitch2note = None
        # self.input2note = None
        # self.note2onset = None
        # self.complex2mag = None
        self.encoder_onset = None
        self.decoder_onset = None

    def encoder_parameters(self):
        """
        Obtain parameters for encoder part of network.

        Returns
        ----------
        parameters : generator
          Layer-wise iterator over parameters
        """

        # Obtain generator for encoder parameters
        parameters = self.encoder.parameters()

        return parameters

    def decoder_parameters(self):
        """
        Obtain parameters for decoder part of network.

        Returns
        ----------
        parameters : generator
          Layer-wise iterator over parameters
        """

        # Obtain generator for decoder parameters
        parameters = self.decoder.parameters()

        return parameters

    def encoder_onset_parameters(self):
        parameters = self.encoder_onset.parameters()

        return parameters

    def decoder_onset_parameters(self):
        parameters = self.decoder_onset.parameters()

        return parameters
    
    # def pitch2note_E_parameters(self):
    #     parameters = self.pitch2note_E.parameters()

    #     return parameters
    
    # def pitch2note_D_parameters(self):
    #     parameters = self.pitch2note_D.parameters()

    #     return parameters
    
    # def note_aug_parameters(self):
    #     parameters = self.note_aug.parameters()

    #     return parameters
    
    # def input2note_parameters(self):
    #     parameters = self.input2note.parameters()

    #     return parameters
    
    # def note2onset_parameters(self):
    #     parameters = self.note2onset.parameters()

    #     return parameters
    
    def get_all_features(self, audio, onset_mode=False, CFP_mode=False):
        """
        Compute all possible features.

        Parameters
        ----------
        audio : Tensor (B x 1 x N)
          Batch of input raw audio
        
        onset_mode : bool
          If True, compute features with one-bin-per-semitone hcqt module

        Returns
        ----------
        features : dict
          Various sets of spectral features
        """

        features_harmonic, features_complex = self.sliCQ(audio)
        
        '''Model input'''
        input = self.sliCQ.to_decibels(features_complex, rescale=False)
        input = self.sliCQ.rescale_decibels(input)

        '''PITCH'''
        # model input
        features_db = self.sliCQ.to_decibels(features_harmonic, rescale=False)
        # Convert decibels to linear probability-like values [0, 1]
        features_am = self.sliCQ.decibels_to_amplitude(features_db)

        # # Convert amplitude to power by squaring
        # features_pw = features_am ** 2

        # Scale decibels to represent probability-like values [0, 1]
        features_db = self.sliCQ.rescale_decibels(features_db)

        # Extract relevant parameters
        harmonics = self.cqt_params['harmonics']
        harmonic_weights = self.cqt_params['weights']

        # Determine first harmonic index
        h_idx = harmonics.index(1)

        # Obtain first harmonic spectral features
        features_db_1 = features_db[:, h_idx]

        # Compute a weighted sum of features to obtain a rough salience estimate
        features_pw_h = torch.sum((features_am * harmonic_weights) ** 2, dim=-3)
        features_db_h = self.sliCQ.to_decibels(features_pw_h ** 0.5, rescale=False)
        features_db_h = self.sliCQ.rescale_decibels(features_db_h)

        # Get CFP
        # audio (B, 1, n_sample) -> audio (B, n_sample)
        if CFP_mode:
          tfrLQ, f, q, t, central_frequencies = self.CFP_HMLC.CFP_filterbank(audio.squeeze(1))
          tfrLQ = tfrLQ.unsqueeze(1)

          features_db = torch.concat([features_db, tfrLQ], dim=1).float()

          tfrLQ = tfrLQ.squeeze(1)
        else:
          tfrLQ = None

        features = {
            'CCQT_db': input, # (B, 2, F(per), T),
            'hcqt': features_db, # model input (B, h+1, F(per), T)
            'pitch_negative_label': features_db_1, # (B, F(per), T)
            'pitch_positive_label': features_db_h,  # (B, F(per), T)

            'tfrLQ': tfrLQ
        }

        if onset_mode:

          features_harmonic, features_complex = self.sliCQ_OPS(audio)

          '''ONSET'''
          # onset label
          features_db_mag_ubdB = features_harmonic[:, h_idx]

          # Compute spectral flux
          # log (1 + |CQT|)
          spectral_flux, spectral_flux_dn, onset_select = self.onset_finder(features_complex)

          # numerical problem ??
          spectral_flux = self.sliCQ.to_decibels(spectral_flux, rescale=False)
          spectral_flux = self.sliCQ_OPS.rescale_decibels(spectral_flux) 

          features = {
            'CCQT_db': input, # model input (B, 2, F(per), T),
            'hcqt': features_db,
            'pitch_negative_label': features_db_1, # (B, F(per), T)
            'pitch_positive_label': features_db_h, # (B, F(per), T)

            'spectral_flux': spectral_flux,
            'onset_selection' : onset_select,

            'tfrLQ': tfrLQ
          }

        return features
    
    def set_model_trainable(self):
        for para in self.parameters():
          para.requires_grad = True

    def set_layer_freeze(self, layer):
        for para in layer.parameters():
          para.requires_grad = False

    def forward(self, features):
        """
        Perform all model functions efficiently (for training/evaluation).

        Parameters
        ----------
        features : Tensor (B x H x F x T)
          Batch of HCQT spectral features

        Returns
        ----------
        output : Tensor (B x F X T)
          Batch of (implicit) pitch salience logits
        ...
        """

        return NotImplementedError

    def save(self, save_path):
        """
        Helper function to save model.

        Parameters
        ----------
        save_path : str
          Path for saving model
        """

        # Pop HCQT module
        cqt = self.sliCQ
        cqt_OPS = self.sliCQ_OPS

        # Pop Onset module
        onset_module = self.onset_finder

        # pop cfp_hmlc module
        cfp_hmlc_module = self.CFP_HMLC

        self.cqt = None
        self.sliCQ_OPS = None
        self.onset_finder = None
        self.CFP_HMLC = None

        if isinstance(self, torch.nn.DataParallel):
            # Unwrap and save the core model
            torch.save(self.module, save_path)
        else:
            # Save the core model
            torch.save(self, save_path)

        # Restore HCQT module
        self.sliCQ = cqt
        self.sliCQ_OPS = cqt_OPS

        # Restore Onset module
        self.onset_finder = onset_module

        # Restore cfp_hmlc module
        self.CFP_HMLC = cfp_hmlc_module

    @staticmethod
    def load(model_path, device='cpu'):
        """
        Helper function to load pre-existing model.

        Parameters
        ----------
        model_path : str
          Path to pre-existing model
        device : str
          Device on which to load model
        """

        # Load a pre-existing model onto specified device
        model = torch.load(model_path, map_location=device)
        
        # Extract stored HCQT parameters
        cqt_params = model.cqt_params.copy()
        cqt_params_OPS = model.cqt_params_OPS.copy()
        CFP_HMLC_params = model.CFP_HMLC_params.copy()

        cqt_params.pop('weights')
        cqt_params_OPS.pop('weights')

        # Re-initialize HQCT module
        model.sliCQ = HCQT(**cqt_params).to(device)
        model.sliCQ_OPS = HCQT(**cqt_params_OPS).to(device)

        # Re-initialize Onset module
        model.onset_finder = Onset_Finder(**cqt_params_OPS).to(device)

        # Re-initialize cfp_hmlc module
        model.CFP_HMLC = CFP_HMLC(**CFP_HMLC_params)

        return model

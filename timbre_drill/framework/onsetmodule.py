# Regular imports
from torchaudio.transforms import AmplitudeToDB
import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import librosa

class Onset_Finder(nn.Module):

    def __init__(self, **kwargs):
        """
        Initialize the full autoencoder.

        Parameters
        ----------
        hcqt_params : dict
          Parameters for feature extraction module
        """

        nn.Module.__init__(self)

        # maximum filter
        self.maxpool_frequency_interval = 3
        self.maxpool_LGD_interval = 3
        self.minpool_LGD_interval = 3

        # onset selection # 10ms each frame
        self.filter_span_front = 9
        self.filter_span_back = 6
        self.average_filter = nn.Conv1d(in_channels=1, out_channels=1,
                                             kernel_size = self.filter_span_front * 2 + 1, stride=1,
                                             bias=False, padding='same')
        
        average_filter_weights = torch.ones_like(self.average_filter.weight)
        average_filter_weights[..., (self.filter_span_front+1+self.filter_span_back):] = 0

        self.average_filter.weight = torch.nn.Parameter(average_filter_weights)
        #self.filter_threshold = 0.24
        self.filter_threshold = 0.48
        self.alphaz = 0.99
    
    @staticmethod
    def to_magnitude(coefficients):
        """
        Compute the magnitude for a set of real coefficients.

        Parameters
        ----------
        coefficients : Tensor (B x 2 x F X T)
          Batch of real/imaginary CQT coefficients

        Returns
        ----------
        magnitude : Tensor (B x F X T)
          Batch of magnitude coefficients
        """

        # Compute L2-norm of coefficients
        magnitude = coefficients.norm(p=2, dim=-3)

        return magnitude

    @staticmethod
    def to_decibels(magnitude, rescale=True):
        """
        Convert magnitude coefficients to decibels.

        TODO - move 0 dB only if maximum is higher?

        Parameters
        ----------
        magnitude : Tensor (B x F X T)
          Batch of magnitude coefficients (amplitude)
        rescale : bool
          Rescale decibels to the range [0, 1]

        Returns
        ----------
        decibels : Tensor (B x F X T)
          Batch of magnitude coefficients (power in dB)
        """

        decibels = list()

        # Loop through each track separately
        for m in magnitude:
            # Convert amplitude coefficients to decibels
            d = AmplitudeToDB(stype='amplitude', top_db=80)(m)

            if rescale:
                # Make ceiling 0 dB
                d -= d.max()
                # Rescale to range [0, 1]
                d = 1 + d / 80

            # Add converted sample to list
            decibels.append(d.unsqueeze(0))

        # Reconstruct original batch
        decibels = torch.cat(decibels, dim=0)

        return decibels

    def to_spectral_flux(self, magnitude):
        """
        Compute the spectral flux.

        Parameters
        ----------
        magnitude : Tensor (B x F X T)
          Batch of magnitude of CQT coefficients

        Returns
        ----------
        spectral flux : Tensor (B x F X T)
          Batch of spectral flux
        """

        spectralFlux = torch.zeros_like(magnitude)

        spectralFlux[..., 1:] = magnitude[..., 1:] - magnitude[..., :-1]

        spectralFlux = F.relu(spectralFlux)

        return spectralFlux
    
    def to_spectral_flux_maxpool(self, magnitude):
        """
        Compute the spectral flux.

        Parameters
        ----------
        magnitude : Tensor (B x F X T)
          Batch of magnitude of CQT coefficients

        Returns
        ----------
        spectral flux : Tensor (B x F X T)
          Batch of spectral flux
        """

        spectralFlux = torch.zeros_like(magnitude)

        # maxpool over frequency
        padding = (self.maxpool_frequency_interval - 1) // 2

        maxpool_magnitude = F.max_pool2d(magnitude, kernel_size=(self.maxpool_frequency_interval, 1), padding=(padding, 0), stride=1)

        spectralFlux[..., 1:] = magnitude[..., 1:] - maxpool_magnitude[..., :-1]

        spectralFlux = F.relu(spectralFlux)

        return spectralFlux
    
    @staticmethod
    def to_angle(complex_feature):
        """
        Compute the spectral flux.

        Parameters
        ----------
        magnitude : Tensor (B x 2 x F X T)
          Batch of magnitude of CQT coefficients

        Returns
        ----------
        spectral flux : Tensor (B x F X T)
          Batch of spectral flux
        """

        complex_number = torch.complex(complex_feature[:, 0, ...], complex_feature[:, 1, ...])
        angle = torch.angle(complex_number)

        return angle
    
    @staticmethod
    def unwrap(angle):
        
        device = angle.device

        np_angle = angle.cpu().detach().numpy()
        unwrap_angle = np.unwrap(np_angle, axis=-2)
        torch_unwrap_angle = torch.from_numpy(unwrap_angle).to(device)

        return torch_unwrap_angle
    
    @staticmethod
    def to_local_group_delay(angle):
        """
        Compute the spectral flux.

        Parameters
        ----------
        magnitude : Tensor (B x F X T)
          Batch of magnitude of CQT coefficients

        Returns
        ----------
        spectral flux : Tensor (B x F X T)
          Batch of spectral flux
        """

        local_group_delay = torch.zeros_like(angle)

        # lowest frequency at [-1]
        local_group_delay[:, :-1, :] = angle[:, :-1, :] - angle[:, 1:, :]

        return local_group_delay
    
    def get_spectral_flux_weight(self, local_group_delay):
        """
        Compute the spectral flux.

        Parameters
        ----------
        magnitude : Tensor (B x F X T)
          Batch of magnitude of CQT coefficients

        Returns
        ----------
        spectral flux : Tensor (B x F X T)
          Batch of spectral flux
        """

        # maxpool over time
        padding = (self.maxpool_LGD_interval - 1) // 2
        spectral_flux_weight = F.max_pool2d(local_group_delay, kernel_size=(1, self.maxpool_LGD_interval), padding=(0, padding), stride=1)

        # minpool over frequency
        padding = (self.minpool_LGD_interval - 1) // 2
        spectral_flux_weight = -F.max_pool2d(-spectral_flux_weight, kernel_size=(self.minpool_LGD_interval, 1), padding=(padding, 0), stride=1)

        return spectral_flux_weight

    def forward(self, complex_feature):
        """
        Compute spectral flux of [0, 1] features and downsample to one bin per semitone

        Parameters
        ----------
        features : Tensor (B x F(per) x T)

        Returns
        ----------
        spectral_flux_dn : Tensor (B x F(one) x T)
        """
        
        features_db_mag = self.to_magnitude(complex_feature) 

        angle = self.to_angle(complex_feature)
        angle = self.unwrap(angle)
        local_group_delay = torch.abs(self.to_local_group_delay(angle))
        weights =  self.get_spectral_flux_weight(local_group_delay)

        # bins_per_semitone = 1
        # new_bins = self.cqt_params['n_bins'] // bins_per_semitone

          # Timestamp(t)' = Timestamp(t) - Timestamp(t-1)

        spectral_flux = self.to_spectral_flux_maxpool(features_db_mag)

          # get average of a semitone
        # spectral_flux_dn = torch.reshape(spectral_flux, (spectral_flux.size(0), bins_per_semitone, new_bins, spectral_flux.size(-1)))
        # spectral_flux_dn = torch.mean(spectral_flux_dn, dim=1) # (B x F(one) x T)

        spectral_flux_dn = spectral_flux * weights
        #spectral_flux_dn = spectral_flux

        '''onset selection'''
        B, K, T = spectral_flux_dn.size()
        device = features_db_mag.device
        spectral_flux_dn_reshape = spectral_flux_dn.view(-1, 1, T) # (B*F, 1, T)

        # criterion 1 
        _, indice_max = F.max_pool1d(spectral_flux_dn_reshape, kernel_size=self.filter_span_back*2+1, padding=self.filter_span_back, stride=1, return_indices=True)
        indice_max = indice_max.view(-1, K, T) # (B*F, 1, T) -> (B, F, T)
        indice_self = torch.arange(T).repeat(B, K, 1).to(device)
        criterion1 = torch.eq(indice_max, indice_self).long()
        
        # criterion 2 
        filter_avg = self.average_filter(spectral_flux_dn_reshape).view(-1, K, T)
        filter_avg = filter_avg / (self.filter_span_front+self.filter_span_back+1) + self.filter_threshold
        criterion2 = torch.gt(spectral_flux_dn, filter_avg).long()

        # result
        result = torch.logical_and(criterion1, criterion2)

        spectral_flux = self.to_spectral_flux(self.to_decibels(features_db_mag))
        
        return spectral_flux, spectral_flux_dn, result.float()
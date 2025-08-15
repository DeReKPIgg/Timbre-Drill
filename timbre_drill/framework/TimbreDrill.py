# Based on Timbre-Trap & SS-MPE # Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

#from timbre_trap.framework import *

from . import SS_NT
from .objectives import *

# Regular imports
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init


class Timbre_Drill(SS_NT):
    """
    Implements base model from Timbre-Trap (https://arxiv.org/abs/2309.15717).
    """

    def __init__(self, cqt_params, CFP_HMLC_params, 
                 onset_bins_per_semitone=1, CFP_mode=False,
                 latent_size=None, model_complexity=1, model_complexity_onset=1, skip_connections=False):
        """
        Initialize the full autoencoder.

        Parameters
        ----------
        See SS_NT class for others...

        latent_size : int or None (Optional)
          Dimensionality of latent space
        model_complexity : int
          Scaling factor for number of filters and embedding sizes
        skip_connections : bool
          Whether to include skip connections between encoder and decoder
        """

        super().__init__(cqt_params, CFP_HMLC_params, onset_bins_per_semitone, CFP_mode)

        n_bins = cqt_params['n_bins']
        bins_per_semitone = cqt_params['bins_per_octave'] // 12
        num_semitones = n_bins // bins_per_semitone
        down_ratio = bins_per_semitone // onset_bins_per_semitone

        if latent_size is None:
            # Set default dimensionality of latents
            latent_size = 32 * 2 ** (model_complexity - 1)

        '''PITCH AUTOENCODER'''

        self.encoder = EncoderNorm(feature_size=n_bins, latent_size=latent_size, model_complexity=model_complexity)
        self.decoder = DecoderNorm(feature_size=n_bins, latent_size=latent_size, model_complexity=model_complexity)

        n_harmonics = len(cqt_params['harmonics'])
        convin_out_channels = self.encoder.convin[0].out_channels

        if CFP_mode:
          encoder_in_dim = n_harmonics+1
        else:
          encoder_in_dim = n_harmonics

        self.encoder.convin = nn.Sequential(
              nn.Conv2d(encoder_in_dim, convin_out_channels, kernel_size=3, padding='same'), # harmonic + CFP
              nn.ELU(inplace=True)
        )

        convout_in_channels = self.decoder.convout[0].in_channels

        self.decoder.convout = nn.Conv2d(convout_in_channels, 1, kernel_size=3, padding='same')

        if skip_connections:
            # Start by adding encoder features with identity weighting
            self.skip_weights = torch.nn.Parameter(torch.ones(5))
        else:
            # No skip connections
            self.skip_weights = None

        '''ONSET AUTOENCODER'''

        self.encoder_onset = EncoderNorm(feature_size=n_bins, latent_size=latent_size, model_complexity=model_complexity_onset)
        self.decoder_onset = DecoderNorm(feature_size=n_bins, latent_size=latent_size, model_complexity=model_complexity_onset)

        onset_convin_out_channels = self.encoder_onset.convin[0].out_channels

        self.encoder_onset.convin = nn.Sequential(
              nn.Conv2d(n_harmonics+1, onset_convin_out_channels, kernel_size=3, padding='same'),
              nn.ELU(inplace=True)
          )

        decoder_convout_in_channels = self.decoder_onset.convout[0].in_channels

        self.decoder_onset.convout = nn.Conv2d(decoder_convout_in_channels, 1, kernel_size=3, padding='same')

        if skip_connections:
            # Start by adding encoder features with identity weighting
            self.skip_weights_onset = torch.nn.Parameter(torch.ones(5))
        else:
            # No skip connections
            self.skip_weights_onset = None

        self.sparsemax = Sparsemax(dim=-2)

    def decoder_parameters(self):
        """
        Obtain parameters for decoder part of network.

        Returns
        ----------
        parameters : generator
          Layer-wise iterator over parameters
        """

        # Obtain parameters corresponding to decoder
        parameters = list(super().decoder_parameters())

        if self.skip_weights is not None:
            # Append skip connection parameters
            parameters.append(self.skip_weights)

        # Return generator type
        for p in parameters:
            yield p
    
    def decoder_onset_parameters(self):
        """
        Obtain parameters for decoder part of network.

        Returns
        ----------
        parameters : generator
          Layer-wise iterator over parameters
        """

        # Obtain parameters corresponding to decoder
        parameters = list(super().decoder_onset_parameters())

        if self.skip_weights_onset is not None:
            # Append skip connection parameters
            parameters.append(self.skip_weights_onset)

        # Return generator type
        for p in parameters:
            yield p

    def apply_skip_connections(self, embeddings):
        """
        Apply skip connections to encoder embeddings, or discard the embeddings if skip connections do not exist.

        Parameters
        ----------
        embeddings : list of [Tensor (B x C x H x T)]
          Embeddings produced by encoder at each level

        Returns
        ----------
        embeddings : list of [Tensor (B x C x H x T)]
          Encoder embeddings scaled with learnable weight
        """

        if self.skip_weights is not None:
            # Apply a learnable weight to the embeddings for the skip connection
            embeddings = [self.skip_weights[i] * e for i, e in enumerate(embeddings)]
        else:
            # Discard embeddings from encoder
            embeddings = None

        return embeddings

    def apply_skip_connections_onset(self, embeddings):
        """
        Apply skip connections to encoder embeddings, or discard the embeddings if skip connections do not exist.

        Parameters
        ----------
        embeddings : list of [Tensor (B x C x H x T)]
          Embeddings produced by encoder at each level

        Returns
        ----------
        embeddings : list of [Tensor (B x C x H x T)]
          Encoder embeddings scaled with learnable weight
        """

        if self.skip_weights_onset is not None:
            # Apply a learnable weight to the embeddings for the skip connection
            embeddings = [self.skip_weights_onset[i] * e for i, e in enumerate(embeddings)]
        else:
            # Discard embeddings from encoder
            embeddings = None

        return embeddings
    
    def decode(self, latents, embeddings=None, transcribe=False):
        """
        Decode a batch of latent codes into logits representing real/imaginary coefficients.

        Parameters
        ----------
        latents : Tensor (B x D_lat x T)
          Batch of latent codes
        embeddings : list of [Tensor (B x C x E x T)] or None (no skip connections)
          Embeddings produced by encoder at each level
        transcribe : bool
          Switch for transcription vs. reconstruction

        Returns
        ----------
        coefficients : Tensor (B x 2 x F X T)
          Batch of output logits [-∞, ∞]
        """

        indicator = (not transcribe) * torch.ones_like(latents[..., :1, :])

        latents_cat = torch.cat((latents, indicator), dim=-2)

        coefficient = self.decoder(latents_cat, embeddings)

        return coefficient

    def decode_onset(self, latents, embeddings=None, transcribe=False):
        """
        Decode a batch of latent codes into logits representing real/imaginary coefficients.

        Parameters
        ----------
        latents : Tensor (B x D_lat x T)
          Batch of latent codes
        embeddings : list of [Tensor (B x C x E x T)] or None (no skip connections)
          Embeddings produced by encoder at each level
        transcribe : bool
          Switch for transcription vs. reconstruction

        Returns
        ----------
        coefficients : Tensor (B x 2 x F X T)
          Batch of output logits [-∞, ∞]
        """

        indicator = (not transcribe) * torch.ones_like(latents[..., :1, :])

        latents_cat = torch.cat((latents, indicator), dim=-2)

        coefficient = self.decoder_onset(latents_cat, embeddings)

        return coefficient

    def forward(self, features, contour_compress=False, transcribe=False, onset_transcribe_only=False, pitch_logits=None):
        """
        Process spectral features to obtain pitch salience logits (for training/evaluation).

        Parameters
        ----------
        features : Tensor (B x 6 x F x T)
          Batch of HCQT spectral features

        Returns
        ----------
        pitch_trans : Tensor (B x 2 x F(per) X T)
          Batch of pitch contour (transcription mode)

        pitch_const : Tensor (B x 2 x F(per) X T)
          Batch of pitch contour (construction mode)

        latents : Tensor (B x D_lat x T)
          Batch of latent codes

        pitch_salience : Tensor (B x F(per) x T)
          Batch of pitch salience logits

        onset_salience : Tensor (B x F(one) x T)
          Batch of onset slience logits

        losses : dict containing
          ...
        """

        pitch_logits = None
        pitch_const  = None
        latents  = None
        pitch_salience = None

        onset_logits = None
        pitch_logits_const = None
        latents_trans = None
        onset_salience = None

        if contour_compress:
          assert(False, "We don't do it here.")

        if ~onset_transcribe_only:
          # timbre-trap 
          latents, embeddings, losses = self.encoder(features)

          embeddings = self.apply_skip_connections(embeddings)

          pitch_logits = self.decode(latents, embeddings, True).squeeze(1) # transcribe

          pitch_const = self.decode(latents, embeddings, False).squeeze(1) # reconstrcut
          
          pitch_salience = ProbLike(pitch_logits)
          
        assert pitch_logits is not None, 'pitch_logits must not be NoneType'
          
        if transcribe or onset_transcribe_only:

          pitch_logits_toAE = torch.cat((features, pitch_logits.unsqueeze(1)), dim=-3)

          latents_trans, embeddings_trans, losses = self.encoder_onset(pitch_logits_toAE)

          embeddings_trans = self.apply_skip_connections_onset(embeddings_trans)

          onset_logits = self.decode_onset(latents_trans, embeddings_trans, True).squeeze(1) # transcribe

          pitch_logits_const = self.decode_onset(latents_trans, embeddings_trans, False).squeeze(1) # reconstrcut

          onset_salience = self.sparsemax(onset_logits.transpose(1, 2)).transpose(1, 2)
          #onset_salience = ProbLike(onset_logits)

        output = {
          'pitch_logits': pitch_logits,
          'pitch_const': pitch_const,
          'latents': latents,
          'pitch_salience': pitch_salience,

          'onset_logits': onset_logits,
          'pitch_logits_const': pitch_logits_const,
          'latents_trans': latents_trans,
          'onset_salience': onset_salience,
        }

        return output
    
    def inference_sep(self, features):

      pitch_logits = torch.zeros(features.size(-2), features.size(-1))
      pitch_salience = torch.zeros(features.size(-2), features.size(-1))
      onset_salience = torch.zeros(features.size(-2), features.size(-1))
        

      '''Pitch Transcription'''

      # 15000 frames ~= 180 secs
      frame_per_chunk = 15000
      overlap_ratio = 0.1

      overlapping = int(frame_per_chunk * overlap_ratio)

      if features.size(-1) % frame_per_chunk == 0:
        chunk_num = (features.size(-1) // frame_per_chunk)
      else:
        chunk_num = (features.size(-1) // frame_per_chunk) + 1

      print("Pitch Transcription\nframe_num: ", features.size(-1), " chunk_num: ", chunk_num)

      for i in range(chunk_num): #  index0: frame_per_chunk / else: frame_per_chunk * overlap_ratio + frame_per_chunk
        if i==0:
          _features = features[..., :frame_per_chunk]
        elif i==chunk_num-1:
          _features = features[..., int((features.size(-1)-1)-frame_per_chunk):]
        else: 
          _features = features[..., int((frame_per_chunk*i)-overlapping):int(frame_per_chunk*(i+1))]
        
        #print("feature shape: ", _features.shape)

        output = self.forward(_features)

        _pitch_logits = output['pitch_logits']
        _pitch_salience = output['pitch_salience']

        if i==0:
          pitch_logits[..., :frame_per_chunk] = _pitch_logits
          pitch_salience[..., :frame_per_chunk] = _pitch_salience
        elif i==chunk_num-1:
          pitch_logits[..., int((features.size(-1)-1)-frame_per_chunk+overlapping):] = _pitch_logits[..., int(overlapping):]
          pitch_salience[..., int((features.size(-1)-1)-frame_per_chunk+overlapping):] = _pitch_salience[..., int(overlapping):]
        else: 
          pitch_logits[..., int(frame_per_chunk*i):int(frame_per_chunk*(i+1))] = _pitch_logits[..., int(overlapping):]
          pitch_salience[..., int(frame_per_chunk*i):int(frame_per_chunk*(i+1))] = _pitch_salience[..., int(overlapping):]

      '''Onset Transcription'''

      # 345 frames ~= 4 secs
      frame_per_chunk = 345
      overlap_ratio = 0.1

      overlapping = int(frame_per_chunk * overlap_ratio)

      if features.size(-1) % frame_per_chunk == 0:
        chunk_num = (features.size(-1) // frame_per_chunk)
      else:
        chunk_num = (features.size(-1) // frame_per_chunk) + 1

      print("Onset Transcription\nframe_num: ", features.size(-1), " chunk_num: ", chunk_num)

      for i in range(chunk_num): #  index0: frame_per_chunk / else: frame_per_chunk * overlap_ratio + frame_per_chunk
        if i==0:
          _features = features[..., :frame_per_chunk]
          _pitch_logits = pitch_logits[..., :frame_per_chunk]
        elif i==chunk_num-1:
          _features = features[..., int((features.size(-1)-1)-frame_per_chunk):]
          _pitch_logits = pitch_logits[..., int((features.size(-1)-1)-frame_per_chunk):]
        else: 
          _features = features[..., int((frame_per_chunk*i)-overlapping):int(frame_per_chunk*(i+1))]
          _pitch_logits = pitch_logits[..., int((frame_per_chunk*i)-overlapping):int(frame_per_chunk*(i+1))]
        
        #print("feature shape: ", _features.shape)

        output = self.forward(_features, onset_transcribe_only=True, pitch_logits=_pitch_logits)

        _onset_salience = output['onset_salience']
        #_onset_salience = output['onset_logits']

        if i==0:
          onset_salience[..., :frame_per_chunk] = _onset_salience
        elif i==chunk_num-1:
          onset_salience[..., int((features.size(-1)-1)-frame_per_chunk+overlapping):] = _onset_salience[..., int(overlapping):]
        else: 
          onset_salience[..., int(frame_per_chunk*i):int(frame_per_chunk*(i+1))] = _onset_salience[..., int(overlapping):]

      return pitch_salience, onset_salience

    def inference(self, features):

      pitch_salience = torch.zeros(features.size(-2), features.size(-1))
      onset_salience = torch.zeros(features.size(-2), features.size(-1))
        
      # 4 secs : 345 frames
      frame_per_chunk = 15000
      overlap_ratio = 0.1

      overlapping = int(frame_per_chunk * overlap_ratio)

      if features.size(-1) % frame_per_chunk == 0:
        chunk_num = (features.size(-1) // frame_per_chunk)
      else:
        chunk_num = (features.size(-1) // frame_per_chunk) + 1

      print("frame_num: ", features.size(-1), " chunk_num: ", chunk_num)

      for i in range(chunk_num): #  index0: frame_per_chunk / else: frame_per_chunk * overlap_ratio + frame_per_chunk
        if i==0:
          _features = features[..., :frame_per_chunk]
        elif i==chunk_num-1:
          _features = features[..., int((features.size(-1)-1)-frame_per_chunk):]
        else: 
          _features = features[..., int((frame_per_chunk*i)-overlapping):int(frame_per_chunk*(i+1))]
        
        #print("feature shape: ", _features.shape)

        output = self.forward(_features, transcribe=True)

        _pitch_salience = output['pitch_salience']
        _onset_salience = output['onset_salience']

        if i==0:
          pitch_salience[..., :frame_per_chunk] = _pitch_salience
          onset_salience[..., :frame_per_chunk] = _onset_salience
        elif i==chunk_num-1:
          pitch_salience[..., int((features.size(-1)-1)-frame_per_chunk+overlapping):] = _pitch_salience[..., int(overlapping):]
          onset_salience[..., int((features.size(-1)-1)-frame_per_chunk+overlapping):] = _onset_salience[..., int(overlapping):]
        else: 
          pitch_salience[..., int(frame_per_chunk*i):int(frame_per_chunk*(i+1))] = _pitch_salience[..., int(overlapping):]
          onset_salience[..., int(frame_per_chunk*i):int(frame_per_chunk*(i+1))] = _onset_salience[..., int(overlapping):]

      return pitch_salience, onset_salience
        

class Encoder(nn.Module):
    """
    Implements a 2D convolutional encoder.
    """

    def __init__(self, feature_size, latent_size=None, model_complexity=1):
        """
        Initialize the encoder.

        Parameters
        ----------
        feature_size : int
          Dimensionality of input features
        latent_size : int or None (Optional)
          Dimensionality of latent space
        model_complexity : int
          Scaling factor for number of filters
        """

        nn.Module.__init__(self)

        channels = (2  * 2 ** (model_complexity - 1),
                    4  * 2 ** (model_complexity - 1),
                    8  * 2 ** (model_complexity - 1),
                    16 * 2 ** (model_complexity - 1),
                    32 * 2 ** (model_complexity - 1))

        # Make sure all channel sizes are integers
        channels = tuple([round(c) for c in channels])

        if latent_size is None:
            # Set default dimensionality
            latent_size = 32 * 2 ** (model_complexity - 1)
              
        self.convin = nn.Sequential(
            nn.Conv2d(2, channels[0], kernel_size=3, padding='same'),
            nn.ELU(inplace=True)
        )

        self.block1 = EncoderBlock(channels[0], channels[1], stride=2)
        self.block2 = EncoderBlock(channels[1], channels[2], stride=2)
        self.block3 = EncoderBlock(channels[2], channels[3], stride=2)
        self.block4 = EncoderBlock(channels[3], channels[4], stride=2)

        embedding_size = feature_size

        for i in range(4):
            # Dimensionality after strided convolutions
            embedding_size = embedding_size // 2 - 1

        self.convlat = nn.Conv2d(channels[4], latent_size, kernel_size=(embedding_size, 1))

    def forward(self, coefficients):
        """
        Encode a batch of input spectral coefficients.

        Parameters
        ----------
        coefficients : Tensor (B x 2 x F X T)
          Batch of spectral coefficients

        Returns
        ----------
        latents : Tensor (B x D_lat x T)
          Batch of latent codes
        embeddings : list of [Tensor (B x C x E x T)]
          Embeddings produced by encoder at each level
        losses : dict containing
          ...
        """

        # Initialize a list to hold features for skip connections
        embeddings = list()

        # Encode features into embeddings
        embeddings.append(self.convin(coefficients))
        embeddings.append(self.block1(embeddings[-1]))
        embeddings.append(self.block2(embeddings[-1]))
        embeddings.append(self.block3(embeddings[-1]))
        embeddings.append(self.block4(embeddings[-1]))

        # Compute latent vectors from embeddings
        latents = self.convlat(embeddings[-1]).squeeze(-2)

        # No encoder losses
        loss = dict()

        return latents, embeddings, loss


class Decoder(nn.Module):
    """
    Implements a 2D convolutional decoder.
    """

    def __init__(self, feature_size, latent_size=None, model_complexity=1):
        """
        Initialize the decoder.

        Parameters
        ----------
        feature_size : int
          Dimensionality of input features
        latent_size : int or None (Optional)
          Dimensionality of latent space
        model_complexity : int
          Scaling factor for number of filters
        """

        nn.Module.__init__(self)

        channels = (32 * 2 ** (model_complexity - 1),
                    16 * 2 ** (model_complexity - 1),
                    8  * 2 ** (model_complexity - 1),
                    4  * 2 ** (model_complexity - 1),
                    2  * 2 ** (model_complexity - 1))

        # Make sure all channel sizes are integers
        channels = tuple([round(c) for c in channels])

        if latent_size is None:
            # Set default dimensionality
            latent_size = 32 * 2 ** (model_complexity - 1)

        padding = list()

        embedding_size = feature_size

        for i in range(4):
            # Padding required for expected output size
            padding.append(embedding_size % 2)
            # Dimensionality after strided convolutions
            embedding_size = embedding_size // 2 - 1

        # Reverse order
        padding.reverse()

        self.convin = nn.Sequential(
            nn.ConvTranspose2d(latent_size + 1, channels[0], kernel_size=(embedding_size, 1)),
            nn.ELU(inplace=True)
        )

        self.block1 = DecoderBlock(channels[0], channels[1], stride=2, padding=padding[0])
        self.block2 = DecoderBlock(channels[1], channels[2], stride=2, padding=padding[1])
        self.block3 = DecoderBlock(channels[2], channels[3], stride=2, padding=padding[2])
        self.block4 = DecoderBlock(channels[3], channels[4], stride=2, padding=padding[3])

        self.convout = nn.Conv2d(channels[4], 2, kernel_size=3, padding='same')

    def forward(self, latents, encoder_embeddings=None):
        """
        Decode a batch of input latent codes.

        Parameters
        ----------
        latents : Tensor (B x D_lat x T)
          Batch of latent codes
        encoder_embeddings : list of [Tensor (B x C x E x T)] or None (no skip connections)
          Embeddings produced by encoder at each level

        Returns
        ----------
        output : Tensor (B x 2 x F X T)
          Batch of output logits [-∞, ∞]
        """

        # Restore feature dimension
        latents = latents.unsqueeze(-2)

        # Process latents with decoder blocks
        embeddings = self.convin(latents)

        if encoder_embeddings is not None:
            embeddings = embeddings + encoder_embeddings[-1]

        embeddings = self.block1(embeddings)

        if encoder_embeddings is not None:
            embeddings = embeddings + encoder_embeddings[-2]

        embeddings = self.block2(embeddings)

        if encoder_embeddings is not None:
            embeddings = embeddings + encoder_embeddings[-3]

        embeddings = self.block3(embeddings)

        if encoder_embeddings is not None:
            embeddings = embeddings + encoder_embeddings[-4]

        embeddings = self.block4(embeddings)

        if encoder_embeddings is not None:
            embeddings = embeddings + encoder_embeddings[-5]

        # Decode embeddings into spectral logits
        output = self.convout(embeddings)

        return output


class EncoderBlock(nn.Module):
    """
    Implements a chain of residual convolutional blocks with progressively
    increased dilation, followed by down-sampling via strided convolution.
    """

    def __init__(self, in_channels, out_channels, stride=2):
        """
        Initialize the encoder block.

        Parameters
        ----------
        in_channels : int
          Number of input feature channels
        out_channels : int
          Number of output feature channels
        stride : int
          Stride for the final convolutional layer
        """

        nn.Module.__init__(self)

        self.block1 = ResidualConv2dBlock(in_channels, in_channels, kernel_size=3, dilation=1)
        self.block2 = ResidualConv2dBlock(in_channels, in_channels, kernel_size=3, dilation=2)
        self.block3 = ResidualConv2dBlock(in_channels, in_channels, kernel_size=3, dilation=3)

        self.hop = stride
        self.win = 2 * stride

        self.sconv = nn.Sequential(
            # Down-sample along frequency (height) dimension via strided convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=(self.win, 1), stride=(self.hop, 1)),
            nn.ELU(inplace=True)
        )

        init.xavier_normal(self.sconv[0].weight, gain=1.0)

    def forward(self, x):
        """
        Feed features through the encoder block.

        Parameters
        ----------
        x : Tensor (B x C_in x H x W)
          Batch of input features

        Returns
        ----------
        y : Tensor (B x C_out x H x W)
          Batch of corresponding output features
        """

        # Process features
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)

        # Down-sample
        y = self.sconv(y)

        return y


class DecoderBlock(nn.Module):
    """
    Implements up-sampling via transposed convolution, followed by a chain
    of residual convolutional blocks with progressively increased dilation.
    """

    def __init__(self, in_channels, out_channels, stride=2, padding=0):
        """
        Initialize the encoder block.

        Parameters
        ----------
        in_channels : int
          Number of input feature channels
        out_channels : int
          Number of output feature channels
        stride : int
          Stride for the transposed convolution
        padding : int
          Number of features to pad after up-sampling
        """

        nn.Module.__init__(self)

        self.hop = stride
        self.win = 2 * stride

        self.tconv = nn.Sequential(
            # Up-sample along frequency (height) dimension via transposed convolution
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(self.win, 1), stride=(self.hop, 1), output_padding=(padding, 0)),
            nn.ELU(inplace=True)
        )

        self.block1 = ResidualConv2dBlock(out_channels, out_channels, kernel_size=3, dilation=1)
        self.block2 = ResidualConv2dBlock(out_channels, out_channels, kernel_size=3, dilation=2)
        self.block3 = ResidualConv2dBlock(out_channels, out_channels, kernel_size=3, dilation=3)

        init.xavier_normal(self.tconv[0].weight, gain=1.0)

    def forward(self, x):
        """
        Feed features through the decoder block.

        Parameters
        ----------
        x : Tensor (B x C_in x H x W)
          Batch of input features

        Returns
        ----------
        y : Tensor (B x C_out x H x W)
          Batch of corresponding output features
        """

        # Up-sample
        y = self.tconv(x)

        # Process features
        y = self.block1(y)
        y = self.block2(y)
        y = self.block3(y)

        return y


class ResidualConv2dBlock(nn.Module):
    """
    Implements a 2D convolutional block with dilation, no down-sampling, and a residual connection.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        """
        Initialize the convolutional block.

        Parameters
        ----------
        in_channels : int
          Number of input feature channels
        out_channels : int
          Number of output feature channels
        kernel_size : int
          Kernel size for convolutions
        dilation : int
          Amount of dilation for first convolution
        """

        nn.Module.__init__(self)

        self.conv1 = nn.Sequential(
            # TODO - only dilate across frequency?
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same', dilation=dilation),
            nn.ELU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ELU(inplace=True)
        )

        init.xavier_normal(self.conv1[0].weight, gain=1.0)
        init.xavier_normal(self.conv2[0].weight, gain=1.0)

    def forward(self, x):
        """
        Feed features through the convolutional block.

        Parameters
        ----------
        x : Tensor (B x C_in x H x W)
          Batch of input features

        Returns
        ----------
        y : Tensor (B x C_out x H x W)
          Batch of corresponding output features
        """

        # Process features
        y = self.conv1(x)
        y = self.conv2(y)

        # Residual connection
        y = y + x

        return y

class EncoderNorm(Encoder):
    """
    Implements the 2D convolutional encoder from Timbre-Trap with layer normalization.
    """

    def __init__(self, feature_size, latent_size=None, model_complexity=1):
        """
        Initialize the encoder.

        Parameters
        ----------
        feature_size : int
          Dimensionality of input features
        latent_size : int or None (Optional)
          Dimensionality of latent space
        model_complexity : int
          Scaling factor for number of filters
        """

        nn.Module.__init__(self)

        channels = (2 * 2 ** (model_complexity - 1),
                    4 * 2 ** (model_complexity - 1),
                    8 * 2 ** (model_complexity - 1),
                    16 * 2 ** (model_complexity - 1),
                    32 * 2 ** (model_complexity - 1))

        # Make sure all channel sizes are integers
        channels = tuple([round(c) for c in channels])

        if latent_size is None:
            # Set default dimensionality
            latent_size = 32 * 2 ** (model_complexity - 1)

        embedding_sizes = [feature_size]

        for i in range(4):
            # Dimensionality after strided convolutions
            embedding_sizes.append(embedding_sizes[-1] // 2 - 1)

        self.convin = nn.Sequential(
            nn.Conv2d(2, channels[0], kernel_size=3, padding='same'),
            nn.ELU(inplace=True),
            LayerNormPermute(normalized_shape=[channels[0], embedding_sizes[0]])
        )

        self.block1 = nn.Sequential(
            EncoderBlock(channels[0], channels[1], stride=2),
            LayerNormPermute(normalized_shape=[channels[1], embedding_sizes[1]])
        )
        self.block2 = nn.Sequential(
            EncoderBlock(channels[1], channels[2], stride=2),
            LayerNormPermute(normalized_shape=[channels[2], embedding_sizes[2]])
        )
        self.block3 = nn.Sequential(
            EncoderBlock(channels[2], channels[3], stride=2),
            LayerNormPermute(normalized_shape=[channels[3], embedding_sizes[3]])
        )
        self.block4 = nn.Sequential(
            EncoderBlock(channels[3], channels[4], stride=2),
            LayerNormPermute(normalized_shape=[channels[4], embedding_sizes[4]])
        )

        self.convlat = nn.Sequential(
            nn.Conv2d(channels[4], latent_size, kernel_size=(embedding_sizes[-1], 1)),
            LayerNormPermute(normalized_shape=[latent_size, 1])
        )

        init.xavier_normal(self.convin[0].weight, gain=1.0)
        init.xavier_normal(self.convlat[0].weight, gain=1.0)


class DecoderNorm(Decoder):
    """
    Implements the 2D convolutional decoder from Timbre-Trap with layer normalization.
    """

    def __init__(self, feature_size, latent_size=None, model_complexity=1):
        """
        Initialize the decoder.

        Parameters
        ----------
        feature_size : int
          Dimensionality of input features
        latent_size : int or None (Optional)
          Dimensionality of latent space
        model_complexity : int
          Scaling factor for number of filters
        """

        nn.Module.__init__(self)

        channels = (32 * 2 ** (model_complexity - 1),
                    16 * 2 ** (model_complexity - 1),
                    8  * 2 ** (model_complexity - 1),
                    4  * 2 ** (model_complexity - 1),
                    2  * 2 ** (model_complexity - 1))

        # Make sure all channel sizes are integers
        channels = tuple([round(c) for c in channels])

        if latent_size is None:
            # Set default dimensionality
            latent_size = 32 * 2 ** (model_complexity - 1)

        padding = list()

        embedding_sizes = [feature_size]

        for i in range(4):
            # Padding required for expected output size
            padding.append(embedding_sizes[-1] % 2)
            # Dimensionality after strided convolutions
            embedding_sizes.append(embedding_sizes[-1] // 2 - 1)

        # Reverse order
        padding.reverse()
        embedding_sizes.reverse()

        self.convin = nn.Sequential(
            nn.ConvTranspose2d(latent_size + 1, channels[0], kernel_size=(embedding_sizes[0], 1)),
            nn.ELU(inplace=True),
            LayerNormPermute(normalized_shape=[channels[0], embedding_sizes[0]])
        )

        self.block1 = nn.Sequential(
            DecoderBlock(channels[0], channels[1], stride=2, padding=padding[0]),
            LayerNormPermute(normalized_shape=[channels[1], embedding_sizes[1]])
        )
        self.block2 = nn.Sequential(
            DecoderBlock(channels[1], channels[2], stride=2, padding=padding[1]),
            LayerNormPermute(normalized_shape=[channels[2], embedding_sizes[2]])
        )
        self.block3 = nn.Sequential(
            DecoderBlock(channels[2], channels[3], stride=2, padding=padding[2]),
            LayerNormPermute(normalized_shape=[channels[3], embedding_sizes[3]])
        )
        self.block4 = nn.Sequential(
            DecoderBlock(channels[3], channels[4], stride=2, padding=padding[3]),
            LayerNormPermute(normalized_shape=[channels[4], embedding_sizes[4]])
        )

        self.convout = nn.Sequential(
            nn.Conv2d(channels[4], 2, kernel_size=3, padding='same'),
            LayerNormPermute(normalized_shape=[2, embedding_sizes[4]])
        )

        init.xavier_normal(self.convin[0].weight, gain=1.0)
        init.xavier_normal(self.convout[0].weight, gain=1.0)
        

class Pitch2NoteEncoder(nn.Module):
    """
    Implements a chain of residual convolutional blocks with progressively
    increased dilation, followed by down-sampling via strided convolution.
    """

    def __init__(self, in_channels, out_channels, down_ratio=3):
        """
        Initialize the encoder block.

        Parameters
        ----------
        in_channels : int
          Number of input feature channels
        out_channels : int
          Number of output feature channels
        stride : int
          Stride for the final convolutional layer
        """

        nn.Module.__init__(self)

        self.block1 = ResidualConv2dBlock_FreqDilated(in_channels, in_channels, kernel_size=3, dilation=1)
        self.block2 = ResidualConv2dBlock_FreqDilated(in_channels, in_channels, kernel_size=3, dilation=2)
        self.block3 = ResidualConv2dBlock_FreqDilated(in_channels, in_channels, kernel_size=3, dilation=3)

        self.hop = down_ratio
        self.win = down_ratio

        self.sconv = nn.Sequential(
            # Down-sample along frequency (height) dimension via strided convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=(self.win, 1), stride=(self.hop, 1)),
            nn.ELU(inplace=True)
        )

        init.xavier_normal(self.sconv[0].weight, gain=1.0)

    def forward(self, x):
        """
        Feed features through the encoder block.

        Parameters
        ----------
        x : Tensor (B x C_in x H x W)
          Batch of input features

        Returns
        ----------
        y : Tensor (B x C_out x H//3 x W)
          Batch of corresponding output features
        """

        # Process features
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)

        # Down-sample
        y = self.sconv(y)

        return y
    
class ResidualConv2dBlock_FreqDilated(nn.Module):
    """
    Implements a 2D convolutional block with dilation, no down-sampling, and a residual connection.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        """
        Initialize the convolutional block.

        Parameters
        ----------
        in_channels : int
          Number of input feature channels
        out_channels : int
          Number of output feature channels
        kernel_size : int
          Kernel size for convolutions
        dilation : int
          Amount of dilation for first convolution
        """

        nn.Module.__init__(self)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same', dilation=(dilation, 1)),
            nn.ELU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ELU(inplace=True)
        )

        init.xavier_normal(self.conv1[0].weight, gain=1.0)
        init.xavier_normal(self.conv2[0].weight, gain=1.0)

    def forward(self, x):
        """
        Feed features through the convolutional block.

        Parameters
        ----------
        x : Tensor (B x C_in x H x W)
          Batch of input features

        Returns
        ----------
        y : Tensor (B x C_out x H x W)
          Batch of corresponding output features
        """

        # Process features
        y = self.conv1(x)
        y = self.conv2(y)

        # Residual connection
        y = y + x

        return y

class LayerNormPermute(nn.LayerNorm):
    """
    Layer normalization with required axis permutation.
    """

    def forward(self, x):
        """
        Feed features through the convolutional block.

        Parameters
        ----------
        x : Tensor (B x C_in x H x W)
          Batch of input features

        Returns
        ----------
        y : Tensor (B x C_out x H x W)
          Batch of corresponding output features
        """

        # Bring channel and feature axis to back
        x = x.permute(0, -1, -3, -2) # layernorm value under same time frame
        # Perform layer normalization
        y = super().forward(x)
        # Restore original dimensionality
        y = y.permute(0, -2, -1, -3)

        return y

class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation

        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.

        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size

        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape and reshape back after sparsemax
        original_size = input.size()
        input = input.contiguous().view(-1, input.size(self.dim))

        dim = 1
        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, device=input.device).view(1, -1)
        range = range.expand_as(zs).type(torch.get_default_dtype())

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        output = self.output.view(original_size)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input

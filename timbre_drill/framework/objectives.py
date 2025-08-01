# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt


__all__ = [
    'compute_sparsity_loss',
    'compute_time_sparsity_loss',

    'compute_support_loss',
    'compute_harmonic_loss',

    'compute_onset_bce_loss',

    'sample_random_equalization',
    'sample_parabolic_equalization',
    'sample_gaussian_equalization',

    'compute_pitch_timbre_loss',
    'compute_onset_timbre_loss',

    'compute_pitch_geometric_loss',
    'compute_onset_geometric_loss',

    'compute_time_sim_loss',
    'compute_frequency_dis_loss',

    'compute_reconstruction_loss',

    'compute_supervised_loss',

    'ProbLike'
]

def ProbLike(activations): 
    #return torch.clamp(F.tanh(activations), min=0, max=1)
    #return torch.clamp(activations, min=0, max=1)
    return F.sigmoid(activations)

def compute_reconstruction_loss(reconstructed, target):
    """
    Compute reconstruction loss for a batch.

    Parameters
    ----------
    reconstructed : Tensor (B x C_in x F X T)
      Batch of reconstructed coefficients
    target : Tensor (B x C_in x F X T)
      Batch of original coefficients

    Returns
    ----------
    reconstruction_loss : tensor (float)
      Total reconstruction loss for the batch
    """

    # Compute mean squared error with respect to every time-frequency bin of coefficients
    reconstruction_loss = torch.nn.functional.mse_loss(reconstructed, target, reduction='none')
    # Sum reconstruction loss across channel / frequency and average across time / batch
    reconstruction_loss = reconstruction_loss.sum(-3).sum(-2).mean()

    return reconstruction_loss

def compute_support_loss(embeddings, h1_features):
    # Set the weight for positive activations to zero
    pos_weight = torch.tensor(0)

    # Compute support loss as BCE of activations with respect to features (negative activations only)
    support_loss = F.binary_cross_entropy_with_logits(embeddings, h1_features, reduction='none', pos_weight=pos_weight)

    # Sum across frequency bins and average across time and batch
    support_loss = support_loss.sum(-2).mean(-1).mean(-1)

    return support_loss

def compute_harmonic_loss(embeddings, salience):
    # Set the weight for negative activations to zero
    neg_weight = torch.tensor(0)

    # Compute harmonic loss as BCE of activations with respect to salience estimate (positive activations only)
    harmonic_loss = F.binary_cross_entropy_with_logits(-embeddings, (1 - salience), reduction='none', pos_weight=neg_weight)

    # Sum across frequency bins and average across time and batch
    harmonic_loss = harmonic_loss.sum(-2).mean(-1).mean(-1)

    return harmonic_loss

def compute_onset_bce_loss(embeddings, onset_detection):
    # Compute bce loss of activations

    pos_weight = torch.tensor(100)

    bce_loss = F.binary_cross_entropy_with_logits(embeddings, onset_detection, reduction='none', pos_weight=pos_weight)

    #bce_loss = F.binary_cross_entropy(embeddings, onset_detection, reduction='none', weight=pos_weight)

    # # positive example

    # pos_loss = -pos_weight * onset_detection * torch.log10(embeddings)

    # # negative example

    # conjugate_label = torch.logical_not(onset_detection).float()
    # ones = torch.ones_like(embeddings)
    # neg_loss = -conjugate_label * torch.log10(ones-embeddings)

    # # Sum across frequency bins and average across time and batch
    # bce_loss = pos_loss + neg_loss
    
    bce_loss = bce_loss.sum(-2).mean(-1).mean(-1)

    return bce_loss

def compute_time_sim_loss(embeddings):
    # (B x F(per) x T)
    embeddings_origin = embeddings[..., :-1]
    embeddings_shift = embeddings[..., 1:]

    # Sum across time bins and average across frequency and batch
    mse_loss = F.mse_loss(embeddings_origin, embeddings_shift, reduction='none').sum(-1).mean()

    return mse_loss

def compute_frequency_dis_loss(embeddings):
    # (B x F(per) x T)
    embeddings_origin = embeddings[:, :-1, :]
    embeddings_shift = embeddings[:, 1:, :]

    mse_loss = F.mse_loss(embeddings_origin, embeddings_shift, reduction='none').sum(-3).sum(-1).mean()

    return -mse_loss

def compute_sparsity_loss(activations):
    # Compute sparsity loss as the L1 norm of the activations
    sparsity_loss = torch.norm(activations, 1, dim=-2)

    # Average loss across time and batch
    sparsity_loss = sparsity_loss.mean(-1).mean(-1)

    return sparsity_loss

def compute_time_sparsity_loss(activations):
    # Compute sparsity loss as the L1 norm of the activations across time
    sparsity_loss = torch.norm(activations, 1, dim=-1)

    # Average loss across time and batch
    sparsity_loss = sparsity_loss.mean(-1).mean(-1)

    return sparsity_loss


def sample_random_equalization(n_bins, batch_size=1, n_points=None, std_dev=0.10, device='cpu'):
    """
    Uniformly sample multiplicative equalization factors and upsample to cover whole frequency spectrum.

    Parameters
    ----------
    n_bins : int
      Final number of frequency bins
    batch_size : int
      Number of curves to sample
    n_points : int or None (optional)
      Number of peaks/troughs to sample
    std_dev : float
      Standard deviation of boost/cut
    device : string
      Device on which to initialize curves

    Returns
    ----------
    curves : Tensor (B x F)
      Sampled equalization curves
    """

    if n_points is None:
        # Default to provided output size
        n_points = n_bins

    # Sample a random equalization curve factor for each sample in batch
    curves = 1 + torch.randn(size=(batch_size, 1, n_points), device=device) * std_dev

    if n_bins != n_points:
        # Upsample equalization curves to number of frequency bins via linear interpolation
        curves = F.interpolate(curves, size=n_bins, mode='linear', align_corners=True)

    # Remove channel dimension
    curves = curves.squeeze(-2)

    return curves


def sample_parabolic_equalization(n_bins, batch_size=1, pointiness=1, device=None):
    """
    Randomly sample parabolic equalization curves covering whole frequency spectrum.

    Parameters
    ----------
    n_bins : int
      Number of frequency bins
    batch_size : int
      Number of curves to sample
    pointiness : float
      Multiplier to shrink parabolic opening
    device : string
      Device on which to initialize curves

    Returns
    ----------
    curves : Tensor (B x F)
      Sampled equalization curves
    """

    # Randomly sample parametric parabolic functions
    alpha, beta = torch.rand(size=(2, batch_size, 1), device=device)
    # Scale parameters to appropriate ranges
    alpha, beta = alpha / (n_bins - 1) ** 2, beta * (n_bins - 1)

    # Create a Tensor of indices for frequency bins of each curve
    idcs = torch.arange(n_bins, device=device).repeat((batch_size, 1))

    # Compute parabolic equalization curves
    curves = 1 - pointiness * alpha * (idcs - beta) ** 2

    return curves


def sample_gaussian_equalization(n_bins, batch_size=1, max_A=0.25, max_std_dev=None, fixed_shape=False, device=None, CFP_mode=False):
    """
    Randomly sample Gaussian equalization curves covering whole frequency spectrum.

    Parameters
    ----------
    n_bins : int
      Number of frequency bins
    batch_size : int
      Number of curves to sample
    max_A : float
      Maximum amplitude of sampled Gaussians
    max_std_dev : float or None (optional)
      Maximum standard deviation of sampled Gaussians
    device : string
      Device on which to initialize curves

    Returns
    ----------
    curves : Tensor (B x F)
      Sampled equalization curves
    """

    if max_std_dev is None:
        # Default to 10% of frequency bins
        max_std_dev = 0.10 * n_bins

    # Randomly sample parametric Gaussian functions
    A, mu, sigma = torch.rand(size=(3, batch_size, 1), device=device)

    if fixed_shape:
        # Amplitude and standard deviation go to maximum values
        A, sigma = A.round(), torch.ones_like(sigma, device=device)

    # Scale parameters to appropriate ranges
    A, mu, sigma = max_A * (A * 2 - 1), mu * (n_bins - 1), sigma * max_std_dev

    # Create a Tensor of indices for frequency bins of each curve
    idcs = torch.arange(n_bins, device=device).repeat((batch_size, 1))

    # Compute Gaussian equalization curves
    curves = 1 + A * torch.exp(-0.5 * (idcs - mu) ** 2 / sigma ** 2)

    return curves


def apply_random_eq(features, hcqt, eq_fn, **eq_kwargs):
    # Obtain dimensionality of features and appropriate device
    (B, H, K, _), device = features.size(), features.device

    # Extract relevant HCQT parameters
    bins_per_octave = hcqt.bins_per_octave

    # Obtain center frequencies (MIDI) associated with each HCQT bin
    midi_freqs = torch.from_numpy(hcqt.midi_freqs).to(device)

    # Infer the number of bins per semitone
    bins_per_semitone = bins_per_octave / 12

    # Determine semitone span of frequency support
    semitone_span = midi_freqs.max() - midi_freqs.min()

    # Determine how many bins are represented across all harmonics
    n_psuedo_bins = (bins_per_semitone * semitone_span).round()

    # Determine how many octaves have been covered
    n_octaves = int(torch.ceil(n_psuedo_bins / bins_per_octave))

    # Perform equalization over full octave
    n_total_bins = n_octaves * bins_per_octave

    # Randomly sample an equalization curve for each sample in batch
    curves = eq_fn(n_total_bins, batch_size=B, device=device, **eq_kwargs)

    # Determine nearest equalization corresponding to each frequency bin
    equalization_bins = bins_per_semitone * (midi_freqs - midi_freqs.min())
    # Round, convert equalization bins to integers, and flatten
    equalization_bins = equalization_bins.round().long().flatten()
    # Obtain indices corresponding to equalization for each sample in the batch
    equalization_idcs = torch.meshgrid(torch.arange(B, device=device), equalization_bins)
    # Obtain the equalization for each sample in the batch

    if eq_kwargs['CFP_mode']:
      equalization = curves[equalization_idcs].view(B, H-1, K, -1) # (H-1): skipping CFP dim 
      # dim=1 is always 1st harmonic
      equalization = torch.concat([equalization, equalization[:, 1, ...].unsqueeze(1)], dim=1)
    else:
      equalization = curves[equalization_idcs].view(B, H, K, -1)  

    # Apply sampled equalization curves to the batch and clamp features
    equalized_features = torch.clip(equalization * features, min=0, max=1)

    return equalized_features


def compute_pitch_timbre_loss(model, features, embeddings, eq_fn, **eq_kwargs):
    # Perform random equalizations on batch of features
    equalized_features = apply_random_eq(features, model.sliCQ, eq_fn, **eq_kwargs)

    # Process equalized features with provided model
    equalization_embeddings = model(equalized_features)['pitch_logits'] # pitch_trans == pitch_logits

    # Convert both sets of logits to activations (implicit pitch salience)
    original_salience, equalization_salience = ProbLike(embeddings), ProbLike(equalization_embeddings)

    # Compute timbre loss as BCE of embeddings computed from equalized features with respect to original activations
    timbre_loss = F.binary_cross_entropy_with_logits(equalization_embeddings, original_salience, reduction='none')

    # Sum across frequency bins and average across time and batch
    timbre_loss = timbre_loss.sum(-2).mean(-1).mean(-1)

    return timbre_loss

def compute_onset_timbre_loss(model, features, embeddings, eq_fn, **eq_kwargs):
    # Perform random equalizations on batch of features
    equalized_features = apply_random_eq(features, model.sliCQ, eq_fn, **eq_kwargs)

    # Process equalized features with provided model
    equalization_embeddings = model(equalized_features, transcribe=True)['onset_logits']

    # Convert both sets of logits to activations (implicit pitch salience)
    original_salience = embeddings

    # Compute timbre loss as BCE of embeddings computed from equalized features with respect to original activations
    timbre_loss = F.binary_cross_entropy_with_logits(equalization_embeddings, original_salience, reduction='none')

    # Sum across frequency bins and average across time and batch
    timbre_loss = timbre_loss.sum(-2).mean(-1).mean(-1)

    return timbre_loss

def apply_translation(tensor, shifts, axis=-1, val=None):
    """
    Perform an independent translation on each entry of a tensor.

    Parameters
    ----------
    tensor : Tensor (B x ...)
      Input tensor with at least 2 dimensions
    shifts : Tensor (B)
      Independent translations to perform
    axis : int
      Axis to translate
    val : float or None (optional)
      Value to insert or None to wrap original data

    Returns
    ----------
    translated : Tensor (B x ...)
      Original tensor translated as specified
    """

    # Determine dimensionality and device of input tensor
    dimensionality, device = tensor.size(), tensor.device

    # Copy original tensor
    tensor = tensor.clone()

    if val is not None:
        # Initialize hidden data to replace wrapped elements
        hidden_data = torch.full(dimensionality, val, device=device)
        # Combine original and hidden data to disable wrapping
        tensor = torch.cat([tensor, hidden_data], dim=axis)

    # Translate each entry by specified amount at specified axis
    translated = torch.cat([x.unsqueeze(0).roll(k.item(), axis)
                            for x, k in zip(tensor, shifts)])

    # Trim translated tensor to original dimensionality
    translated = translated.narrow(axis, 0, dimensionality[axis])

    return translated


def apply_distortion(tensor, stretch_factors):
    """
    Perform an independent distortion on each entry of a tensor.

    Parameters
    ----------
    tensor : Tensor (B x C x H x W)
      Input tensor with standard 4 dimensions
    stretch_factors : Tensor (B)
      Independent distortions to perform

    Returns
    ----------
    distorted : Tensor (B x ...)
      Original tensor distorted as specified
    """

    # Initialize list for distortions
    distorted = list()

    # Loop through each entry and scale
    for x, t in zip(tensor, stretch_factors):
        # Stretch entry by specified factor using linear interpolation
        distorted_ = F.interpolate(x, scale_factor=t.item(), mode='linear')

        if t >= 1:
            # Determine starting index to center distortion
            start_idx = (distorted_.size(-1) - x.size(-1)) // 2
            # Center distorted tensor and trim to original width
            distorted_ = distorted_.narrow(-1, start_idx, x.size(-1))
        else:
            # Determine total padding necessary
            pad_t = x.size(-1) - distorted_.size(-1)
            # Distribute padding between both sides
            pad_l, pad_r = pad_t // 2, pad_t - pad_t // 2
            # Pad distorted tensor to match original width
            distorted_ = F.pad(distorted_, (pad_l, pad_r))

        # Append distorted entry to distortion list
        distorted.append(distorted_.unsqueeze(0))

    # Combine all distorted entries
    distorted = torch.cat(distorted)

    return distorted


def compute_pitch_geometric_loss(model, features, embeddings, max_shift_v, max_shift_h, max_stretch_factor):
    # Determine batch size
    B = features.size(0)

    # Sample a random vertical / horizontal shift for each sample in the batch
    v_shifts = torch.randint(low=-max_shift_v, high=max_shift_v + 1, size=(B,))
    h_shifts = torch.randint(low=-max_shift_h, high=max_shift_h + 1, size=(B,))

    # Compute inverse of maximum stretch factor
    min_stretch_factor = 1 / max_stretch_factor

    # Sample a random stretch factor for each sample in the batch, starting at minimum
    stretch_factors, stretch_factors_ = min_stretch_factor, torch.rand(size=(B,))
    # Split sampled values into piecewise ranges
    neg_perc = 2 * stretch_factors_.clip(max=0.5)
    pos_perc = 2 * (stretch_factors_ - 0.5).relu()
    # Scale stretch factor evenly across effective range
    stretch_factors += neg_perc * (1 - min_stretch_factor)
    stretch_factors += pos_perc * (max_stretch_factor - 1)

    # Apply vertical and horizontal translations, inserting zero at empties
    transformed_features = apply_translation(features, v_shifts, axis=-2, val=0)
    transformed_features = apply_translation(transformed_features, h_shifts, axis=-1, val=0)
    # Apply time distortion, maintaining original dimensionality and padding with zeros
    transformed_features = apply_distortion(transformed_features, stretch_factors)

    # Process transformed features with provided model
    # pitch_trans, pitch_const, latents, pitch_logits, pitch_salience, note_salience, onset_logits, onset_salience, losses
    transformation_embeddings = model(transformed_features)['pitch_logits']

    # Convert logits to activations (implicit pitch salience)
    salience = ProbLike(embeddings).unsqueeze(-3)

    # Apply same transformations to activations produced for original features
    transformed_salience = apply_translation(salience, v_shifts, axis=-2, val=0)
    transformed_salience = apply_translation(transformed_salience, h_shifts, axis=-1, val=0)
    transformed_salience = apply_distortion(transformed_salience, stretch_factors)

    # Remove temporarily added channel dimension
    transformed_salience = transformed_salience.squeeze(-3)

    # Compute geometric loss as BCE of embeddings computed from transformed features with respect to transformed activations
    geometric_loss = F.binary_cross_entropy_with_logits(transformation_embeddings, transformed_salience, reduction='none')

    # Sum across frequency bins and average across time and batch
    geometric_loss = geometric_loss.sum(-2).mean(-1).mean(-1)

    return geometric_loss

def compute_onset_geometric_loss(model, features, embeddings, max_shift_v, max_shift_h, max_stretch_factor):
    # Determine batch size
    B = features.size(0)

    # Sample a random vertical / horizontal shift for each sample in the batch
    v_shifts = torch.randint(low=-max_shift_v, high=max_shift_v + 1, size=(B,))
    h_shifts = torch.randint(low=-max_shift_h, high=max_shift_h + 1, size=(B,))

    # Compute inverse of maximum stretch factor
    min_stretch_factor = 1 / max_stretch_factor

    # Sample a random stretch factor for each sample in the batch, starting at minimum
    stretch_factors, stretch_factors_ = min_stretch_factor, torch.rand(size=(B,))
    # Split sampled values into piecewise ranges
    neg_perc = 2 * stretch_factors_.clip(max=0.5)
    pos_perc = 2 * (stretch_factors_ - 0.5).relu()
    # Scale stretch factor evenly across effective range
    stretch_factors += neg_perc * (1 - min_stretch_factor)
    stretch_factors += pos_perc * (max_stretch_factor - 1)

    # Apply vertical and horizontal translations, inserting zero at empties
    transformed_features = apply_translation(features, v_shifts, axis=-2, val=0)
    transformed_features = apply_translation(transformed_features, h_shifts, axis=-1, val=0)
    # Apply time distortion, maintaining original dimensionality and padding with zeros
    transformed_features = apply_distortion(transformed_features, stretch_factors)

    # Process transformed features with provided model
    # pitch_trans, pitch_const, latents, pitch_logits, pitch_salience, note_salience, onset_logits, onset_salience, losses
    transformed_embeddings = model(transformed_features, transcribe=True)['onset_logits']

    # Convert logits to activations (implicit pitch salience)
    salience = embeddings.unsqueeze(-3)

    # Apply same transformations to activations produced for original features
    transformed_salience = apply_translation(salience, v_shifts, axis=-2, val=0)
    transformed_salience = apply_translation(transformed_salience, h_shifts, axis=-1, val=0)
    transformed_salience = apply_distortion(transformed_salience, stretch_factors)

    # Remove temporarily added channel dimension
    transformed_salience = transformed_salience.squeeze(-3)

    # Compute geometric loss as BCE of embeddings computed from transformed features with respect to transformed activations
    geometric_loss = F.binary_cross_entropy_with_logits(transformed_embeddings, transformed_salience, reduction='none')

    # Sum across frequency bins and average across time and batch
    geometric_loss = geometric_loss.sum(-2).mean(-1).mean(-1)

    return geometric_loss


def compute_supervised_loss(embeddings, ground_truth):
    # Compute supervised loss as BCE of activations with respect to ground-truth
    supervised_loss = F.binary_cross_entropy_with_logits(embeddings, ground_truth, reduction='none')

    # Sum across frequency bins and average across time and batch
    supervised_loss = supervised_loss.sum(-2).mean(-1).mean(-1)

    return supervised_loss

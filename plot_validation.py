from timbre_drill.datasets.MixedMultiPitch import URMP as URMP_Mixtures, Bach10 as Bach10_Mixtures, Su, TRIOS, MAPS
from timbre_drill.datasets.SoloMultiPitch import GuitarSet, URMP as URMP_Stems
from timbre_drill.datasets import ComboDataset

from timbre_drill.datasets.SoloMultiPitch import NSynth

from timbre_drill.framework import *
from timbre_drill.framework.objectives import *
from timbre_drill.utils import *
from evaluate import evaluate
from evaluate_onset import evaluate_onset

from torch.utils.tensorboard import SummaryWriter
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from sacred import Experiment
from tqdm import tqdm

import numpy as np
import warnings
import librosa
import torch
import torch.nn.functional as F
import math
import os

import matplotlib.pyplot as plt

######################
## TRAINING SETTING ##
######################

DEBUG = 1 # (0 - off | 1 - on)
CONFIG = 1 # (0 - MCT server | 1 - PC)
EX_NAME = '_'.join(['Local_testing'])

ex = Experiment('Validate a model to perform NT and plot result')

@ex.config
def config():
    ##############################
    ## TRAINING HYPERPARAMETERS ##
    ##############################

    # Specify a checkpoint from which to resume training (None to disable)
    checkpoint_path = "/root/MCT/checkpoint/raw_clamp/model-11400.pt"

    # Maximum number of training iterations to conduct
    max_epochs = 300

    # Number of iterations between checkpoints
    checkpoint_interval = 250 # 250

    # Number of samples to gather for a batch
    batch_size = 1 if DEBUG else 20 # 20

    # Number of seconds of audio per sample
    n_secs = 4

    # Initial learning rate for encoder
    learning_rate_encoder = 1e-4

    # Initial learning rate for decoder
    learning_rate_decoder = learning_rate_encoder

    learning_rate_pitch2note = learning_rate_encoder * 0.1
    learning_rate_input2note = learning_rate_encoder * 0.1
    learning_rate_note2onset = learning_rate_encoder * 0.1

    # Group together both learning rates
    learning_rates = [learning_rate_encoder, learning_rate_decoder, learning_rate_pitch2note, learning_rate_input2note, learning_rate_note2onset]

    # Scaling factors for each loss term
    multipliers = {
        'support_p' : 1.2,
        'harmonic_p' : 1.5,
        'sparsity_p' : 1.5,
        'timbre_p' : 1,
        'geometric_p' : 1,
        'bce_o' : 1,
        'sparsity_t_o' : 0.5,
        'sparsity_f_o' : 0.5,
        'timbre_o' : 1,
        'geometric_o' : 1,
        'reconstruction' : 1,
        'supervised' : 0
    }

    # Number of epochs spanning warmup phase (0 to disable)
    n_epochs_warmup = 0

    # Set validation dataset to compare for learning rate decay and early stopping
    validation_criteria_set = MAPS.name()

    # Set validation metric to compare for learning rate decay and early stopping
    validation_criteria_metric = 'loss/total'

    # Select whether the validation criteria should be maximized or minimized
    validation_criteria_maximize = False # (False - minimize | True - maximize)

    # Late starting point (0 to disable)
    n_epochs_late_start = 0

    # Number of epochs without improvement before reducing learning rate (0 to disable)
    n_epochs_decay = 2

    # Number of epochs before starting epoch counter for learning rate decay
    n_epochs_cooldown = 0

    # Number of epochs without improvement before early stopping (None to disable)
    n_epochs_early_stop = None

    # IDs of the GPUs to use, if available
    gpu_ids = [0] if DEBUG else [3]

    # Random seed for this experiment
    seed = 42

    ########################
    ## FEATURE EXTRACTION ##
    ########################

    '''sliCQ'''
    # Number of samples per second of audio
    sample_rate = 22050

    # Number of samples between frames
    hop_length = 256

    # # First center frequency (MIDI) of geometric progression
    fmin = librosa.note_to_midi('A0')

    # Number of bins in a single octave
    bins_per_octave = 60 # 5 bins per semitone

    # Number of octaves the CQT should span
    n_octaves = 9

    '''hcqt'''
    harmonics = [0.5, 1, 2, 3, 4, 5]

    '''onset'''
    onset_bins_per_semitone = 1


    ############
    ## OTHERS ##
    ############

    # Number of threads to use for data loading
    n_workers = 0 if DEBUG else 16 * len(gpu_ids)

    # Top-level directory under which to save all experiment files
    root_dir = os.path.join('generated', 'experiments', EX_NAME)

    # Create the root directory
    os.makedirs(root_dir, exist_ok=True)

    if DEBUG:
        # Print a warning message indicating debug mode is active
        warnings.warn('Running in DEBUG mode...', RuntimeWarning)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))

@ex.automain
def get_model_transcription(checkpoint_path, max_epochs, checkpoint_interval, batch_size, n_secs, learning_rates,
                multipliers, n_epochs_warmup, validation_criteria_set, validation_criteria_metric,
                validation_criteria_maximize, n_epochs_late_start, n_epochs_decay, n_epochs_cooldown,
                n_epochs_early_stop, gpu_ids, seed, sample_rate, hop_length, fmin, bins_per_octave,
                n_octaves, harmonics, n_workers, root_dir, onset_bins_per_semitone):
    # Discard read-only types
    learning_rates = list(learning_rates)
    multipliers = dict(multipliers)
    harmonics = list(harmonics)
    gpu_ids = list(gpu_ids)

    # Seed everything with the same seed
    seed_everything(seed)

    # Initialize the primary PyTorch device
    device = torch.device(f'cuda:{gpu_ids[0]}'
                          if torch.cuda.is_available() else 'cpu')

    ########################
    ## FEATURE EXTRACTION ##
    ########################

    n_bins = n_octaves * bins_per_octave
    
    # Create weighting for harmonics (harmonic loss)
    harmonic_weights = 1 / torch.Tensor(harmonics) ** 2
    # Apply zero weight to sub-harmonics (harmonic loss)
    harmonic_weights[harmonic_weights > 1] = 0
    # Normalize the harmonic weights
    harmonic_weights /= torch.sum(harmonic_weights)
    # Add frequency and time dimensions for broadcasting
    harmonic_weights = harmonic_weights.unsqueeze(-1).unsqueeze(-1)
    # Make sure weights are on appropriate device
    harmonic_weights = harmonic_weights.to(device)

    hcqt_params = {'sample_rate': sample_rate,
                   'hop_length': hop_length,
                   'fmin': fmin,
                   'bins_per_octave': bins_per_octave,
                   'n_bins': n_bins,
                   'gamma': None,
                   'harmonics': harmonics,
                   'weights' : harmonic_weights}
    
    # Determine maximum supported MIDI frequency
    fmax = fmin + n_bins / (bins_per_octave / 12)

    # Load best model and make sure it is in evaluation mode
    if checkpoint_path is None:
        # Initialize model
        model = Timbre_Drill(hcqt_params,
                        latent_size=128,
                        model_complexity=2,
                        skip_connections=True,
                        onset_bins_per_semitone=onset_bins_per_semitone)
    else:
        model = SS_NT.load(checkpoint_path, device=device)

    model.eval()

    # Point to the datasets within the storage drive containing them or use the default location
    nsynth_base_dir    = None if CONFIG else "/host/data/datasets/NSynth"
    mnet_base_dir      = None if CONFIG else None
    mydb_base_dir      = None if CONFIG else None
    magna_base_dir     = None if CONFIG else None
    fma_base_dir       = None if CONFIG else None
    mydb_ptch_base_dir = None if CONFIG else None
    urmp_base_dir      = None if CONFIG else "/host/data/datasets/URMP"
    bch10_base_dir     = None if CONFIG else None
    gset_base_dir      = None if CONFIG else None
    mstro_base_dir     = None if CONFIG else None
    swd_base_dir       = None if CONFIG else None
    su_base_dir        = None if CONFIG else None
    trios_base_dir     = None if CONFIG else None
    maps_base_dir = None if CONFIG else '/host/data/datasets/MAPS'

    urmp_train_splits = URMP_Mixtures.available_splits()

    maps_train_splits = ['MAPS_AkPnBcht_2', 
                  'MAPS_AkPnBsdf_2',
                  'MAPS_AkPnCGdD_2',
                  'MAPS_AkPnStgb_2',
                  'MAPS_ENSTDkCl_2',
                  'MAPS_SptkBGAm_2']
    
    maps_val_splits = ['MAPS_SptkBGCl_2', 'MAPS_StbgTGd2_2']
    
    maps_test_splits = ['MAPS_ENSTDkAm_2', 'MAPS_ENSTDkCl_2']

    # Initialize list to hold all training datasets
    pitch_train = list()
    onset_train = list()

    if DEBUG:
        # Instantiate NSynth validation split for training
        nsynth_stems_train = NSynth(base_dir=nsynth_base_dir,
                                    splits=['valid'],
                                    n_tracks=200,
                                    midi_range=np.array([fmin, fmax]),
                                    sample_rate=sample_rate,
                                    cqt=model.sliCQ,
                                    n_secs=n_secs,
                                    seed=seed)
        pitch_train.append(nsynth_stems_train)

        MAPS_train = MAPS(base_dir=maps_base_dir,
                 splits=maps_test_splits,
                 sample_rate=sample_rate,
                 cqt=model.sliCQ_OPS,
                 n_secs=n_secs,
                 seed=seed)
        
        onset_train.append(MAPS_train)
    else:
        # Instantiate NSynth training split for training

        nsynth_stems_train = NSynth(base_dir=nsynth_base_dir,
                                    splits=['train'],
                                    midi_range=np.array([fmin, fmax]),
                                    sample_rate=sample_rate,
                                    cqt=model.sliCQ,
                                    n_secs=n_secs,
                                    seed=seed)
        pitch_train.append(nsynth_stems_train)

        # urmp_mixes_train = URMP_Mixtures(base_dir=urmp_base_dir,
        #                                  splits=urmp_train_splits,
        #                                  sample_rate=sample_rate,
        #                                  cqt=model.sliCQ,
        #                                  n_secs=n_secs,
        #                                  seed=seed)
        # all_train.append(urmp_mixes_train)

        # urmp_stems_train = URMP_Stems(base_dir=urmp_base_dir,
        #                               splits=urmp_train_splits,
        #                               sample_rate=sample_rate,
        #                               cqt=model.sliCQ,
        #                               n_secs=n_secs,
        #                               seed=seed)
        #all_train.append(urmp_stems_train)

        MAPS_train = MAPS(base_dir=maps_base_dir,
                 splits=maps_train_splits,
                 sample_rate=sample_rate,
                 cqt=model.sliCQ,
                 n_secs=n_secs,
                 seed=seed)
        
        onset_train.append(MAPS_train)


    
    # Combine all training datasets
    pitch_train = ComboDataset(pitch_train)
    onset_train = ComboDataset(onset_train)

    # Initialize a PyTorch dataloader
    pitch_loader = DataLoader(dataset=pitch_train,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=n_workers,
                        pin_memory=True,
                        drop_last=True)
    
    if len(onset_train):

        to_onset_loader = onset_train
        origin_step_per_loader = len(onset_train) // batch_size

        if origin_step_per_loader < checkpoint_interval:
            augment_step = checkpoint_interval // origin_step_per_loader + 1
            for i in range(augment_step):
                to_onset_loader = ConcatDataset([to_onset_loader, onset_train])
        
        # Initialize a PyTorch dataloader for audio data
        onset_loader = DataLoader(dataset=to_onset_loader,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=n_workers,
                        pin_memory=True,
                        drop_last=True)
    else:
        # Replace dataloader with null list
        onset_loader = [None] * len(pitch_loader)

    # Instantiate NSynth validation split for validation
    nsynth_val = NSynth(base_dir=nsynth_base_dir,
                        splits=['valid'],
                        n_tracks=200,
                        midi_range=np.array([fmin, fmax]),
                        sample_rate=sample_rate,
                        cqt=model.sliCQ,
                        seed=seed)
    
    # Set the URMP validation set as was defined in the MT3 paper
    urmp_val_splits = ['01', '02', '12', '13', '24', '25', '31', '38', '39']
    # Instantiate URMP dataset mixtures for validation
    urmp_val = URMP_Mixtures(base_dir=urmp_base_dir,
                             splits=urmp_val_splits,
                             sample_rate=sample_rate,
                             cqt=model.sliCQ,
                             seed=seed)

    # Instantiate NSynth testing split for evaluation
    nsynth_test = NSynth(base_dir=nsynth_base_dir,
                         splits=['test'],
                         sample_rate=sample_rate,
                         cqt=model.sliCQ,
                         seed=seed)

    # Instantiate URMP dataset mixtures for evaluation
    urmp_test = URMP_Mixtures(base_dir=urmp_base_dir,
                              splits=None,
                              sample_rate=sample_rate,
                              cqt=model.sliCQ,
                              seed=seed)

    # Add all validation datasets to a list
    # validation_sets = [nsynth_val, urmp_val, bch10_test, su_test, trios_test]
    validation_sets = [nsynth_val, urmp_val]

    # Add all evaluation datasets to a list
    # evaluation_sets = [nsynth_test, bch10_test, su_test, trios_test, urmp_test, gset_test]
    #evaluation_sets = [nsynth_test, urmp_test]
    evaluation_sets = [urmp_test]

    #################
    ## PREPARATION ##
    #################

    # Maximum amplitude for Gaussian equalization
    max_A = 0.375

    # Maximum standard deviation for Gaussian equalization
    max_std_dev = 2 * bins_per_octave

    # Whether to sample fixed rather than varied shapes
    fixed_shape = False

    # Set keyword arguments for Gaussian equalization
    gaussian_kwargs = {
        'max_A' : max_A,
        'max_std_dev' : max_std_dev,
        'fixed_shape' : fixed_shape
    }

    # Set equalization type and corresponding parameter values
    eq_fn, eq_kwargs = sample_gaussian_equalization, gaussian_kwargs

    ####################
    ## GEOMETRIC LOSS ##
    ####################

    # Determine training sequence length in frames
    n_frames = int(n_secs * sample_rate / hop_length)

    # Define maximum time and frequency shift
    max_shift_v = 2 * bins_per_octave
    max_shift_h = n_frames // 4

    # Maximum rate by which audio can be sped up or slowed down
    max_stretch_factor = 2

    # Set keyword arguments for geometric transformations
    gm_kwargs = {
        'max_shift_v' : max_shift_v,
        'max_shift_h' : max_shift_h,
        'max_stretch_factor' : max_stretch_factor
    }

    ################
    ## EVALUATION ##
    ################

    # if cudnn_benchmarking:
    #     # Disable benchmarking prior to evaluation
    #     torch.backends.cudnn.benchmark = False

    for eval_set in evaluation_sets:
        # Evaluate the model using testing split
        final_results = evaluate(model=model,
                                 eval_set=eval_set,
                                 multipliers=multipliers,
                                 device=device,
                                 eq_fn=eq_fn,
                                 eq_kwargs=eq_kwargs,
                                 gm_kwargs=gm_kwargs,
                                 plotNsave=True)
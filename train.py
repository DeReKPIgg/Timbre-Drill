from timbre_drill.datasets.MixedMultiPitch import URMP as URMP_Mixtures, Bach10 as Bach10_Mixtures, Su, TRIOS, MAPS, MusicNet
from timbre_drill.datasets.SoloMultiPitch import GuitarSet, URMP as URMP_Stems
from timbre_drill.datasets import ComboDataset

from timbre_drill.datasets.SoloMultiPitch import NSynth

from timbre_drill.framework import *
from timbre_drill.framework.objectives import *
from timbre_drill.utils import *
from evaluate import evaluate
from evaluate_note import evaluate_note
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

DEBUG = 0 # (0 - off | 1 - on)
CONFIG = 0 # (0 - MCT server | 1 - PC)
EX_NAME = '_'.join(['Local_testing'])

ex = Experiment('Train a model to perform NT with self-supervised objectives only')

@ex.config
def config():
    ##############################
    ## TRAINING HYPERPARAMETERS ##
    ##############################

    TRAIN_FROM_SCRATCH = False
    Phase1 = False
    Phase2 = False
    Phase3 = True

    pitch_set = 'Nsynth' # Nsynth
    onset_set = 'MusicNet' # MAPS or URMP or MusicNet

    set_dict = {'Nsynth': NSynth.name(), 'MAPS': MAPS.name(), 'URMP': URMP_Mixtures.name(), 'MusicNet': MusicNet.name()}

    # Specify a checkpoint from which to resume training (None to disable)

    checkpoint_path = '/host/data/experiments_result/TD-10-2(2AE)/MusicNet/pitch/model-24300.pt'
    #checkpoint_path = None

    # Maximum number of training iterations to conduct
    max_epochs = 1 if CONFIG else 100

    # Number of iterations between checkpoints
    checkpoint_interval = 1 if CONFIG else 300

    # Number of samples to gather for a batch
    batch_size = 1 if DEBUG else 20 # 20
    
    if ~DEBUG and Phase3:
        batch_size = 16

    # Number of seconds of audio per sample
    n_secs = 4

    # Initial learning rate for encoder
    learning_rate_encoder = 1e-4

    # Initial learning rate for decoder
    learning_rate_decoder = learning_rate_encoder

    # Initial learning rate for note & onset
    learning_rate_onset_encoder = learning_rate_encoder * 5
    learning_rate_onset_decoder = learning_rate_encoder * 5

    # Group together both learning rates
    learning_rates = [learning_rate_encoder, learning_rate_decoder, learning_rate_onset_encoder, learning_rate_onset_decoder]

    # Scaling factors for each loss term
    multipliers = {
        'support_p' : 1.2,
        'harmonic_p' : 1.5,
        'sparsity_p' : 1.5,
        'timbre_p' : 1,
        'geometric_p' : 1,
        'time_sim': 1.5,

        'sparsity_n': 5,
        'note_harmonic': 0.1,
        'note_support': 1,
        'note_compress': 5,
        'frequency_distance': 0,

        'bce_o' : 5,
        'sparsity_t_o' : 1,
        'sparsity_f_o' : 0,
        'timbre_o' : 1,
        'geometric_o' : 1,

        'reconstruction' : 1,
        'reconstruction_o' : 1,

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
    gpu_ids = [0] if DEBUG else [2]

    # Random seed for this experiment
    seed = 4200

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
    n_octaves = 8

    '''hcqt'''
    harmonics = [0.5, 1, 2, 3, 4, 5]

    '''onset'''
    onset_bins_per_semitone = 5

    '''CFP_HMLC'''
    CFP_HMLC_win = 7412
    CFP_HMLC_fr = 1.0
    CFP_HMLC_g = np.array([0.2, 0.6, 0.9, 0.9, 1.0])
    CFP_HMLC_bov = 2
    CFP_HMLC_Har = 1
    CFP_mode = False

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
def train_model(TRAIN_FROM_SCRATCH, Phase1, Phase2, Phase3, pitch_set, onset_set, set_dict,
                checkpoint_path, max_epochs, checkpoint_interval, batch_size, n_secs, learning_rates,
                multipliers, n_epochs_warmup, validation_criteria_set, validation_criteria_metric,
                validation_criteria_maximize, n_epochs_late_start, n_epochs_decay, n_epochs_cooldown,
                n_epochs_early_stop, gpu_ids, seed, sample_rate, hop_length, fmin, bins_per_octave,
                n_octaves, harmonics, onset_bins_per_semitone,
                CFP_mode,
                CFP_HMLC_win, CFP_HMLC_fr, CFP_HMLC_g, CFP_HMLC_bov, CFP_HMLC_Har,
                n_workers, root_dir):
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
    
    fmin = librosa.midi_to_hz(fmin)

    CFP_HMLC_params = { 'fs': sample_rate,
                        'win': CFP_HMLC_win,
                        'hop': hop_length,
                        'fc': fmin,
                        'tc': 1/(fmin*(2**n_octaves)),
                        'NumPerOctave': bins_per_octave,
                        'fr': CFP_HMLC_fr,
                        'g': CFP_HMLC_g,
                        'bov': CFP_HMLC_bov,
                        'Har': CFP_HMLC_Har
    }
    
    # Determine maximum supported MIDI frequency
    fmax = fmin + n_bins / (bins_per_octave / 12)

    if checkpoint_path is None:
        # Initialize autoencoder model
        model = Timbre_Drill(cqt_params=hcqt_params,
                        CFP_HMLC_params=CFP_HMLC_params,
                        latent_size=128,
                        model_complexity=2,
                        model_complexity_onset=1,
                        skip_connections=True,
                        onset_bins_per_semitone=onset_bins_per_semitone,
                        CFP_mode=CFP_mode)
    else:
        # Load weights of the specified model checkpoint
        model = SS_NT.load(checkpoint_path, device=device)

    if len(gpu_ids) > 1:
        # Wrap model for multi-GPU usage
        model = DataParallel(model, device_ids=gpu_ids)

    model = model.to(device)

    # Point to the datasets within the storage drive containing them or use the default location
    nsynth_base_dir    = None if CONFIG else "/host/dataset/NSynth"
    mnet_base_dir      = None if CONFIG else None
    mydb_base_dir      = None if CONFIG else None
    magna_base_dir     = None if CONFIG else None
    fma_base_dir       = None if CONFIG else None
    mydb_ptch_base_dir = None if CONFIG else None
    urmp_base_dir      = None if CONFIG else "/host/dataset/URMP"
    bch10_base_dir     = None if CONFIG else None
    gset_base_dir      = None if CONFIG else None
    mstro_base_dir     = None if CONFIG else None
    swd_base_dir       = None if CONFIG else None
    su_base_dir        = None if CONFIG else None
    trios_base_dir     = None if CONFIG else None
    maps_base_dir = None if CONFIG else '/host/dataset/MAPS'
    musicnet_base_dir = None if CONFIG else '/host/dataset/MusicNet'

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
    all_train = list()
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
        all_train.append(nsynth_stems_train)
    else:
        # Instantiate NSynth training split for training
        nsynth_stems_train = NSynth(base_dir=nsynth_base_dir,
                                    splits=['train'],
                                    midi_range=np.array([fmin, fmax]),
                                    sample_rate=sample_rate,
                                    cqt=model.sliCQ,
                                    n_secs=n_secs,
                                    seed=seed)
        all_train.append(nsynth_stems_train)

    # Combine all training datasets
    all_train = ComboDataset(all_train)

    # Initialize a PyTorch dataloader
    loader = DataLoader(dataset=all_train,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=n_workers,
                        pin_memory=True,
                        drop_last=True)
                    
    # Instantiate NSynth validation split for validation
    nsynth_val = NSynth(base_dir=nsynth_base_dir,
                        splits=['valid'],
                        n_tracks=200,
                        midi_range=np.array([fmin, fmax]),
                        sample_rate=sample_rate,
                        cqt=model.sliCQ,
                        seed=seed)

    urmp_mixes_train = URMP_Mixtures(base_dir=urmp_base_dir,
                                         splits=urmp_train_splits,
                                         sample_rate=sample_rate,
                                         cqt=model.sliCQ,
                                         n_secs=n_secs,
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

    MAPS_train = MAPS(base_dir=maps_base_dir,
                 splits=maps_train_splits,
                 sample_rate=sample_rate,
                 cqt=model.sliCQ,
                 n_secs=n_secs,
                 seed=seed)
    
    MAPS_val = MAPS(base_dir=maps_base_dir,
                 splits=maps_val_splits,
                 sample_rate=sample_rate,
                 cqt=model.sliCQ,
                 seed=seed)

    MAPS_test = MAPS(base_dir=maps_base_dir,
                 splits=maps_test_splits,
                 sample_rate=sample_rate,
                 cqt=model.sliCQ,
                 seed=seed)

    MusicNet_train = MusicNet(base_dir=musicnet_base_dir,
                 splits=['train'],
                 sample_rate=sample_rate,
                 cqt=model.sliCQ,
                 n_secs=n_secs,
                 seed=seed)

    MusicNet_val = MusicNet(base_dir=musicnet_base_dir,
                 splits=['test'],
                 sample_rate=sample_rate,
                 cqt=model.sliCQ,
                 seed=seed)

    if onset_set == 'MAPS':
        onset_train.append(MAPS_train)
    elif onset_set == 'URMP':
        onset_train.append(urmp_mixes_train)
    elif onset_set == 'MusicNet':
        onset_train.append(MusicNet_train)
    else:
        assert(False, "Invalid onset training set.")

    onset_train = ComboDataset(onset_train)

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

    # Add all validation datasets to a list
    # validation_sets = [nsynth_val, urmp_val, bch10_test, su_test, trios_test]
    validation_sets = [nsynth_val, urmp_val, MAPS_val, MusicNet_val]
    validation_sets_onset = [urmp_val, MAPS_val, MusicNet_val]

    # Add all evaluation datasets to a list
    # evaluation_sets = [nsynth_test, bch10_test, su_test, trios_test, urmp_test, gset_test]
    evaluation_sets = [nsynth_val, nsynth_test, urmp_val, urmp_test, MusicNet_val, MAPS_val, MAPS_test]

    #################
    ## PREPARATION ##
    #################

    # Initialize an optimizer for the model parameters with differential learning rates
    optimizer = torch.optim.AdamW([{'params' : model.encoder_parameters(), 'lr' : learning_rates[0]},
                                   {'params' : model.decoder_parameters(), 'lr' : learning_rates[1]}])
    
    # optimizer_note = torch.optim.AdamW([
    #                             {'params' : model.pitch2note_E_parameters(), 'lr' : learning_rates[2]},
    #                             {'params' : model.pitch2note_D_parameters(), 'lr' : learning_rates[3]}])

    # optimizer_onset = torch.optim.AdamW([
    #                             {'params' : model.note_aug_parameters(), 'lr' : learning_rates[4]},
    #                             {'params' : model.input2note_parameters(), 'lr' : learning_rates[5]},
    #                             {'params' : model.note2onset_parameters(), 'lr' : learning_rates[6]}])

    optimizer_onset = torch.optim.AdamW([
                                {'params' : model.encoder_onset_parameters(), 'lr' : learning_rates[2]},
                                {'params' : model.decoder_onset_parameters(), 'lr' : learning_rates[3]}])

    # Determine amount of batches in one epoch
    # onset train is shorter
    epoch_steps = len(onset_train)

    # Compute number of validation checkpoints corresponding to learning rate decay cooldown and window
    n_checkpoints_cooldown = math.ceil(n_epochs_cooldown * epoch_steps / checkpoint_interval)
    n_checkpoints_decay = math.ceil(n_epochs_decay * epoch_steps / checkpoint_interval)

    if n_epochs_early_stop is not None:
        # Compute number of validation checkpoints corresponding to early stopping window
        n_checkpoints_early_stop = math.ceil(n_epochs_early_stop * epoch_steps / checkpoint_interval)
    else:
        # Early stopping is disabled
        n_checkpoints_early_stop = None

    # Warmup global learning rate over a fixed number of steps according to a cosine function
    warmup_scheduler = CosineWarmup(optimizer, n_steps=n_epochs_warmup * checkpoint_interval)
    #warmup_scheduler_note = CosineWarmup(optimizer_note, n_steps=n_epochs_warmup * checkpoint_interval)
    warmup_scheduler_onset = CosineWarmup(optimizer_onset, n_steps=n_epochs_warmup * checkpoint_interval)

    # Decay global learning rate by a factor of 1/2 after validation performance has plateaued
    decay_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                 mode='max' if validation_criteria_maximize else 'min',
                                                                 factor=0.5,
                                                                 patience=n_checkpoints_decay,
                                                                 threshold=2E-3,
                                                                 cooldown=n_checkpoints_cooldown)
    
    # decay_scheduler_note = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_note,
    #                                                              mode='max' if validation_criteria_maximize else 'min',
    #                                                              factor=0.5,
    #                                                              patience=n_checkpoints_decay,
    #                                                              threshold=2E-3,
    #                                                              cooldown=n_checkpoints_cooldown)

    decay_scheduler_onset = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_onset,
                                                                 mode='max' if validation_criteria_maximize else 'min',
                                                                 factor=0.5,
                                                                 patience=n_checkpoints_decay,
                                                                 threshold=2E-3,
                                                                 cooldown=n_checkpoints_cooldown)
    
    # Enable anomaly detection to debug any NaNs (can increase overhead)
    # torch.autograd.set_detect_anomaly(True)

    # Enable cuDNN auto-tuner to optimize CUDA kernel (might improve
    # performance, but adds initial overhead to find the best kernel)
    cudnn_benchmarking = False

    if cudnn_benchmarking:
        # Enable benchmarking prior to training
        torch.backends.cudnn.benchmark = True

    # Construct the path to the directory for saving models
    log_dir = os.path.join(root_dir, 'models')

    # Initialize a writer to log results
    writer = SummaryWriter(log_dir)

    # Number of batches that have been processed
    batch_count = 0

    # Keep track of the model with the best validation results
    best_model_checkpoint = None

    # Keep track of the best model's results for comparison
    best_results = None

    # Counter for number of checkpoints since previous best results
    n_checkpoints_elapsed = 0

    # Flag to indicate early stopping criteria has been met
    early_stop_criteria = False

    #################
    ## TIMBRE LOSS ##
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
        'fixed_shape' : fixed_shape,
        'CFP_mode' : CFP_mode
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

    ##############################
    ## TRAINING/VALIDATION LOOP ##
    ##############################
    if TRAIN_FROM_SCRATCH:
        Phase1 = True
        Phase2 = False
        Phase3 = False
        mode_num = 3
    else:
        if Phase1 and not Phase2 and not Phase3:
            mode_num = 1
        elif Phase1 and Phase2 and not Phase3:
            mode_num = 2
        elif not Phase1 and Phase2 and not Phase3:
            mode_num = 1
        elif not Phase1 and Phase2 and Phase3:
            mode_num = 2
        elif not Phase1 and not Phase2 and Phase3:
            mode_num = 1
        else:
            assert(False, "Invalid phase combination.")
    
    # Loop through epochs
    for i in range(max_epochs*mode_num):

        if i == max_epochs and mode_num==3:
            Phase1 = False
            Phase2 = True
            Phase3 = False
        elif i == max_epochs and mode_num==2 and Phase1:
            Phase1 = False
            Phase2 = True
            Phase3 = False
        elif i == max_epochs and mode_num==2 and Phase2:
            Phase1 = False
            Phase2 = False
            Phase3 = True
        elif i == max_epochs*2 and mode_num==3:
            Phase1 = False
            Phase2 = False
            Phase3 = True
            break

        if Phase1:

            #Phase1_loader = loader

            Phase1_loader = onset_loader
            
            loader_multiplier = 1

            # Loop through batches of audio
            for data in tqdm(Phase1_loader, desc=f'Epoch {i + 1} pitch'):
                # Increment the batch counter
                batch_count += 1

                if warmup_scheduler.is_active():
                    # Step the learning rate warmup scheduler
                    warmup_scheduler.step()

                # Extract audio and add to appropriate device
                audio = data[constants.KEY_AUDIO].to(device)
                ground_truth = torch.Tensor(data[constants.KEY_GROUND_TRUTH])

                # Log the current learning rates for this batch
                writer.add_scalar('train/loss/learning_rate/encoder', optimizer.param_groups[0]['lr'], batch_count)
                writer.add_scalar('train/loss/learning_rate/decoder', optimizer.param_groups[1]['lr'], batch_count)

                # Compute full set of spectral features
                features = model.get_all_features(audio, CFP_mode=CFP_mode)

                # Extract relevant feature sets
                input = features['hcqt'] # (B, 2, F(per), T)
                weak_label_neg = features['pitch_negative_label'] # (B, F(per), T)
                weak_label_pos = features['pitch_positive_label'] # (B, F(per), T)

                # tfrL0 = features['tfrL0']
                # tfrLF = features['tfrLF']
                # tfrLQ = features['tfrLQ']

                with torch.autocast(device_type=f'cuda'):
                    
                    #############
                    ## Phase 1 ##
                    #############
                    '''------layer setting------'''
                    model.set_model_trainable()
                    model.set_layer_freeze(model.encoder_onset)
                    model.set_layer_freeze(model.decoder_onset)
                    '''------layer setting------'''

                    output = model(input)

                    pitch_logits = output['pitch_logits']
                    pitch_const = output['pitch_const']
                    pitch_salience = output['pitch_salience']

                    # RECONSTRUCTION
                    reconstruction_loss = compute_reconstruction_loss(weak_label_neg, pitch_const)
                    writer.add_scalar('train/loss/reconstruction', reconstruction_loss.item(), batch_count)

                    # PITCH SPARSITY
                    pitch_sparsity_loss = compute_sparsity_loss(pitch_salience)
                    writer.add_scalar('train/loss/pitch_sparsity', pitch_sparsity_loss.item(), batch_count)

                    # PITCH BINARY CROSS ENTROPY
                    pitch_support_loss = compute_support_loss(pitch_logits, weak_label_neg)
                    writer.add_scalar('train/loss/pitch_support', pitch_support_loss.item(), batch_count)
                    pitch_harmonic_loss = compute_harmonic_loss(pitch_logits, weak_label_pos)
                    writer.add_scalar('train/loss/pitch_harmonic', pitch_harmonic_loss.item(), batch_count)

                    # PITCH TIMBRE 
                    pitch_timbre_loss = compute_pitch_timbre_loss(model, input, pitch_logits, eq_fn, **eq_kwargs)
                    writer.add_scalar('train/loss/pitch_timbre', pitch_timbre_loss.item(), batch_count)
                    # PITCH GEOMETRIC
                    pitch_geometric_loss = compute_pitch_geometric_loss(model, input, pitch_logits, **gm_kwargs)
                    writer.add_scalar('train/loss/pitch_geometric', pitch_geometric_loss.item(), batch_count)

                    # TIME SIMILARITY
                    pitch_time_sim_loss = compute_time_sim_loss(pitch_salience)
                    writer.add_scalar('train/loss/pitch_time_similarity', pitch_time_sim_loss.item(), batch_count)

                    Phase1_loss = multipliers['support_p'] * pitch_support_loss + \
                                multipliers['harmonic_p'] * pitch_harmonic_loss + \
                                multipliers['sparsity_p'] * pitch_sparsity_loss + \
                                multipliers['timbre_p'] * pitch_timbre_loss + \
                                multipliers['geometric_p'] * pitch_geometric_loss + \
                                multipliers['reconstruction'] * reconstruction_loss + \
                                multipliers['time_sim'] * pitch_time_sim_loss
                    
                    Phase1_loss *= loader_multiplier
                    
                    optimizer.zero_grad()
                    Phase1_loss.backward()

                    # Compute the average gradient norm across the encoder
                    avg_norm_encoder = average_gradient_norms(model.encoder)
                    # Log the average gradient norm of the encoder for this batch
                    writer.add_scalar('Phase1/avg_norm/encoder', avg_norm_encoder, batch_count)
                    # Determine the maximum gradient norm across encoder
                    max_norm_encoder = get_max_gradient_norm(model.encoder)
                    # Log the maximum gradient norm of the encoder for this batch
                    writer.add_scalar('Phase1/max_norm/encoder', max_norm_encoder, batch_count)

                    # Compute the average gradient norm across the decoder
                    avg_norm_decoder = average_gradient_norms(model.decoder)
                    # Log the average gradient norm of the decoder for this batch
                    writer.add_scalar('Phase1/avg_norm/decoder', avg_norm_decoder, batch_count)
                    # Determine the maximum gradient norm across decoder
                    max_norm_decoder = get_max_gradient_norm(model.decoder)
                    # Log the maximum gradient norm of the decoder for this batch
                    writer.add_scalar('Phase1/max_norm/decoder', max_norm_decoder, batch_count)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

                    optimizer.step()

                    if batch_count % checkpoint_interval == 0:
                         # Construct a path to save the model checkpoint
                        model_path = os.path.join(log_dir, f'model-{batch_count}.pt')
                        # Save model checkpoint
                        model.save(model_path)

                        if cudnn_benchmarking:
                            # Disable benchmarking prior to validation
                            torch.backends.cudnn.benchmark = False

                        # Initialize dictionary to hold all validation results
                        validation_results = dict()

                        for val_set in validation_sets:
                            # Validate the model checkpoint on each validation dataset
                            validation_results[val_set.name()] = evaluate(model=model,
                                                                        eval_set=val_set,
                                                                        multipliers=multipliers,
                                                                        THRESHOLD=0.5,
                                                                        writer=writer,
                                                                        i=batch_count,
                                                                        device=device,
                                                                        eq_fn=eq_fn,
                                                                        eq_kwargs=eq_kwargs,
                                                                        gm_kwargs=gm_kwargs)

                        # Make sure model is on correct device and switch to training mode
                        model = model.to(device)
                        model.train()

                        if cudnn_benchmarking:
                            # Re-enable benchmarking after validation
                            torch.backends.cudnn.benchmark = True

                        if decay_scheduler.patience and not warmup_scheduler.is_active() and i >= n_epochs_late_start:
                            # Step the learning rate decay scheduler by logging the validation metric for the checkpoint
                            decay_scheduler.step(validation_results[validation_criteria_set][validation_criteria_metric])

                        # Extract the result on the specified metric from the validation results for comparison
                        current_score = validation_results[validation_criteria_set][validation_criteria_metric]

                        if best_results is not None:
                            # Extract the currently tracked best result on the specified metric for comparison
                            best_score = best_results[validation_criteria_set][validation_criteria_metric]

                        if best_results is None or \
                                (validation_criteria_maximize and current_score > best_score) or \
                                (not validation_criteria_maximize and current_score < best_score):
                            print(f'New best at {batch_count} iterations...')

                            # Set current checkpoint as best
                            best_model_checkpoint = batch_count
                            # Update best results
                            best_results = validation_results
                            # Reset number of checkpoints
                            n_checkpoints_elapsed = 0
                        else:
                            # Increment number of checkpoints
                            n_checkpoints_elapsed += 1

                        if n_checkpoints_early_stop is not None and n_checkpoints_elapsed >= n_checkpoints_early_stop:
                            # Early stop criteria has been reached
                            early_stop_criteria = True

                            break

                        if early_stop_criteria:
                            # Stop training
                            break
        else:
            print('No Phase1 training...')

        # if Phase2:
        #     for data in tqdm(onset_loader, desc=f'Epoch {i + 1} note'):
        #         # Increment the batch counter
        #         batch_count += 1    

        #         if warmup_scheduler_note.is_active():
        #             # Step the learning rate warmup scheduler
        #             warmup_scheduler_note.step()

        #         audio = data[constants.KEY_AUDIO].to(device)
        #         ground_truth = torch.Tensor(data[constants.KEY_GROUND_TRUTH])
        #         features = model.get_all_features(audio, CFP_mode=CFP_mode)

        #         # Extract relevant feature sets
        #         input = features['hcqt'] # (B, 2, F(per), T)
        #         #onset_select = features['onset_selection']
        #         weak_label_neg = features['pitch_negative_label'] # (B, F(per), T)
        #         weak_label_pos = features['pitch_positive_label'] # (B, F(per), T)

        #         # Log the current learning rates for this batch
        #         writer.add_scalar('train/LR/learning_rate/pitch2note_E', optimizer_note.param_groups[0]['lr'], batch_count)
        #         writer.add_scalar('train/LR/learning_rate/pitch2note_D', optimizer_note.param_groups[1]['lr'], batch_count)

        #         with torch.autocast(device_type=f'cuda'):
        #             #############
        #             ## Phase 2 ##
        #             #############
        #             model.set_model_trainable()
        #             model.set_layer_freeze(model.encoder)
        #             model.set_layer_freeze(model.decoder)
        #             model.set_layer_freeze(model.note_aug)
        #             model.set_layer_freeze(model.input2note)
        #             model.set_layer_freeze(model.note2onset)

        #             _, _, latents, _, _, note_logits, note_salience, _, _, losses = model(input, contour_compress=True)

        #             # NOTE HARMONIC
        #             note_harmonic_loss = compute_harmonic_loss(note_logits, weak_label_pos)
        #             writer.add_scalar('train/loss/note_harmonic', note_harmonic_loss.item(), batch_count)

        #             # NOTE SUPPORT
        #             note_support_loss = compute_support_loss(note_logits, weak_label_neg)
        #             writer.add_scalar('train/loss/note_support', note_support_loss.item(), batch_count)
                    
        #             # NOTE SPARSITY
        #             note_sparsity_loss = compute_sparsity_loss(note_salience)
        #             writer.add_scalar('train/loss/note_freq_sparsity', note_sparsity_loss.item(), batch_count)

        #             # NOTE COMPRESS
        #             note_time_sim_loss = compute_time_sim_loss(note_salience)
        #             writer.add_scalar('train/loss/time_compress', note_time_sim_loss.item(), batch_count)

        #             # NOTE DISTANCE
        #             note_frequency_dis_loss = compute_frequency_dis_loss(note_salience)
        #             writer.add_scalar('train/loss/frequency_distance', note_frequency_dis_loss.item(), batch_count)

        #             Phase2_loss = multipliers['note_compress'] * note_time_sim_loss + \
        #                             multipliers['note_harmonic'] * note_harmonic_loss + \
        #                             multipliers['note_support'] * note_support_loss + \
        #                             multipliers['frequency_distance'] * note_frequency_dis_loss + \
        #                             multipliers['sparsity_n'] * note_sparsity_loss
                    
        #             optimizer_note.zero_grad()
        #             Phase2_loss.backward()

        #             # Compute the average gradient norm across the pitch2note encoder
        #             avg_norm_decoder = average_gradient_norms(model.pitch2note_E)
        #             # Log the average gradient norm of the decoder for this batch
        #             writer.add_scalar('Phase2/avg_norm/pitch2note_E', avg_norm_decoder, batch_count)
        #             # Determine the maximum gradient norm across pitch2note encoder
        #             max_norm_decoder = get_max_gradient_norm(model.pitch2note_E)
        #             # Log the maximum gradient norm of the decoder for this batch
        #             writer.add_scalar('Phase2/max_norm/pitch2note_E', max_norm_decoder, batch_count)

        #             # Compute the average gradient norm across the pitch2note decoder
        #             avg_norm_decoder = average_gradient_norms(model.pitch2note_D)
        #             # Log the average gradient norm of the decoder for this batch
        #             writer.add_scalar('Phase2/avg_norm/pitch2note_D', avg_norm_decoder, batch_count)
        #             # Determine the maximum gradient norm across pitch2note decoder
        #             max_norm_decoder = get_max_gradient_norm(model.pitch2note_D)
        #             # Log the maximum gradient norm of the decoder for this batch
        #             writer.add_scalar('Phase2/max_norm/pitch2note_D', max_norm_decoder, batch_count)

        #             # Apply gradient clipping for training stability
        #             torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        #             # Perform an optimization step
        #             optimizer_note.step()

        #         if batch_count % checkpoint_interval == 0:
        #              # Construct a path to save the model checkpoint
        #             model_path = os.path.join(log_dir, f'model-{batch_count}.pt')
        #             # Save model checkpoint
        #             model.save(model_path)

        #             if cudnn_benchmarking:
        #                 # Disable benchmarking prior to validation
        #                 torch.backends.cudnn.benchmark = False

        #             # Initialize dictionary to hold all validation results
        #             validation_results = dict()

        #             for val_set in validation_sets:
        #                 # Validate the model checkpoint on each validation dataset
        #                 validation_results[val_set.name()] = evaluate_note(model=model,
        #                                                             eval_set=val_set,
        #                                                             multipliers=multipliers,
        #                                                             THRESHOLD=0.5,
        #                                                             writer=writer,
        #                                                             i=batch_count,
        #                                                             device=device,
        #                                                             eq_fn=eq_fn,
        #                                                             eq_kwargs=eq_kwargs,
        #                                                             gm_kwargs=gm_kwargs)

        #             # Make sure model is on correct device and switch to training mode
        #             model = model.to(device)
        #             model.train()

        #             if cudnn_benchmarking:
        #                 # Re-enable benchmarking after validation
        #                 torch.backends.cudnn.benchmark = True

        #                 if decay_scheduler_note.patience and not warmup_scheduler_note.is_active() and i >= n_epochs_late_start:
        #                     # Step the learning rate decay scheduler by logging the validation metric for the checkpoint
        #                     decay_scheduler_note.step(validation_results[validation_criteria_set][validation_criteria_metric])

        #                 # Extract the result on the specified metric from the validation results for comparison
        #                 current_score = validation_results[validation_criteria_set][validation_criteria_metric]

        #                 if best_results is not None:
        #                     # Extract the currently tracked best result on the specified metric for comparison
        #                     best_score = best_results[validation_criteria_set][validation_criteria_metric]

        #                 if best_results is None or \
        #                         (validation_criteria_maximize and current_score > best_score) or \
        #                         (not validation_criteria_maximize and current_score < best_score):
        #                     print(f'New best at {batch_count} iterations...')

        #                     # Set current checkpoint as best
        #                     best_model_checkpoint = batch_count
        #                     # Update best results
        #                     best_results = validation_results
        #                     # Reset number of checkpoints
        #                     n_checkpoints_elapsed = 0
        #                 else:
        #                     # Increment number of checkpoints
        #                     n_checkpoints_elapsed += 1

        #                 if n_checkpoints_early_stop is not None and n_checkpoints_elapsed >= n_checkpoints_early_stop:
        #                     # Early stop criteria has been reached
        #                     early_stop_criteria = True

        #                     break

        #                 if early_stop_criteria:
        #                     # Stop training
        #                     break
        # else:
        #     print('No Phase2 training...')

        if Phase3:
            for data in tqdm(onset_loader, desc=f'Epoch {i + 1} onset'):
                # Increment the batch counter
                batch_count += 1    

                if warmup_scheduler_onset.is_active():
                    # Step the learning rate warmup scheduler
                    warmup_scheduler_onset.step()

                audio = data[constants.KEY_AUDIO].to(device)
                ground_truth = torch.Tensor(data[constants.KEY_GROUND_TRUTH])
                features = model.get_all_features(audio, onset_mode=True, CFP_mode=CFP_mode)

                # Extract relevant feature sets
                input = features['hcqt'] # (B, 2, F(per), T)

                spectral_flux = features['spectral_flux']
                onset_select = features['onset_selection']

                weak_label_neg = features['pitch_negative_label'] # (B, F(per), T)
                #weak_label_pos = features['pitch_positive_label'] # (B, F(per), T)

                # Log the current learning rates for this batch
                writer.add_scalar('train/LR/learning_rate/encoder_onset', optimizer_onset.param_groups[0]['lr'], batch_count)
                writer.add_scalar('train/LR/learning_rate/decoder_onset', optimizer_onset.param_groups[1]['lr'], batch_count)


                with torch.autocast(device_type=f'cuda'):
                    #############
                    ## Phase 3 ##
                    #############
                    model.set_model_trainable()
                    model.set_layer_freeze(model.encoder)
                    model.set_layer_freeze(model.decoder)

                    # output = {
                    #     'pitch_logits': pitch_logits,
                    #     'pitch_const': pitch_const,
                    #     'latents': latents,
                    #     'pitch_salience': pitch_salience,

                    #     'onset_logits': onset_logits,
                    #     'pitch_logits_const': pitch_logits_const,
                    #     'latents_trans': latents_trans,
                    #     'onset_salience': onset_salience,

                    #     'HCQT_logits': HCQT_logits,
                    #     'pitch_const_const': pitch_const_const,
                    #     'latents_const': latents_const
                    # }

                    output = model(input, transcribe=True)

                    pitch_logits = output['pitch_logits']
                    pitch_const = output['pitch_const']
                    pitch_salience = output['pitch_salience']

                    # print(pitch_logits)
                    # print(pitch_const)
                    # print(pitch_salience)

                    onset_logits = output['onset_logits']
                    pitch_logits_const = output['pitch_logits_const']
                    onset_salience = output['onset_salience']

                    # print(onset_logits)
                    # print(pitch_logits_const)
                    # print(onset_salience)

                    '''pitch trans / onset const & pitch const / onset const''' 
                    onset_reconstruction_loss = compute_reconstruction_loss(spectral_flux, pitch_logits_const)
                    writer.add_scalar('train/loss/onset_reconstruction', onset_reconstruction_loss.item(), batch_count)

                    '''pitch trans / onset trans'''
                    
                    # ONSET SPARSITY
                    onset_frequency_sparsity_loss = compute_sparsity_loss(onset_salience)
                    writer.add_scalar('train/loss/onset_freq_sparsity', onset_frequency_sparsity_loss.item(), batch_count)

                    onset_time_sparsity_loss = compute_time_sparsity_loss(ProbLike(onset_logits))
                    writer.add_scalar('train/loss/onset_time_sparsity', onset_time_sparsity_loss.item(), batch_count)

                    # ONSET BINARY CROSS ENTROPY
                    onset_bce_loss = compute_onset_bce_loss(onset_logits, onset_select)
                    writer.add_scalar('train/loss/onset_bce', onset_bce_loss.item(), batch_count)
                    # ONSET TIMBRE
                    onset_timbre_loss = compute_onset_timbre_loss(model, input, onset_salience, eq_fn, **eq_kwargs)
                    writer.add_scalar('train/loss/onset_timbre', onset_timbre_loss.item(), batch_count)
                    # ONSET GEOMETRIC
                    onset_geometric_loss = compute_onset_geometric_loss(model, input, onset_salience, **gm_kwargs)
                    writer.add_scalar('train/loss/onset_geometric', onset_geometric_loss.item(), batch_count)

                    Phase3_loss = multipliers['bce_o'] * onset_bce_loss + \
                                    multipliers['sparsity_f_o'] * onset_frequency_sparsity_loss + \
                                    multipliers['sparsity_t_o'] * onset_time_sparsity_loss + \
                                    multipliers['timbre_o'] * onset_timbre_loss + \
                                    multipliers['geometric_o'] * onset_geometric_loss + \
                                    multipliers['reconstruction_o'] * onset_reconstruction_loss
                    
                    optimizer_onset.zero_grad()
                    Phase3_loss.backward()

                    # Compute the average gradient norm across the encoder_onset
                    avg_norm_encoder = average_gradient_norms(model.encoder_onset)
                    # Log the average gradient norm of the encoder for this batch
                    writer.add_scalar('Phase3/avg_norm/encoder_onset', avg_norm_encoder, batch_count)
                    # Determine the maximum gradient norm across encoder_onset
                    max_norm_encoder = get_max_gradient_norm(model.encoder_onset)
                    # Log the maximum gradient norm of the encoder for this batch
                    writer.add_scalar('Phase3/max_norm/encoder_onset', max_norm_encoder, batch_count)

                    # Compute the average gradient norm across the decoder_onset
                    avg_norm_decoder = average_gradient_norms(model.decoder_onset)
                    # Log the average gradient norm of the decoder for this batch
                    writer.add_scalar('Phase3/avg_norm/decoder_onset', avg_norm_decoder, batch_count)
                    # Determine the maximum gradient norm across decoder_onset
                    max_norm_decoder = get_max_gradient_norm(model.decoder_onset)
                    # Log the maximum gradient norm of the decoder for this batch
                    writer.add_scalar('Phase3/max_norm/decoder_onset', max_norm_decoder, batch_count)

                    # Apply gradient clipping for training stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                    # Perform an optimization step
                    optimizer_onset.step()

                if batch_count % checkpoint_interval == 0:
                     # Construct a path to save the model checkpoint
                    model_path = os.path.join(log_dir, f'{onset_set}_{batch_count}.pt')
                    # Save model checkpoint
                    model.save(model_path)

                    if cudnn_benchmarking:
                        # Disable benchmarking prior to validation
                        torch.backends.cudnn.benchmark = False

                    # Initialize dictionary to hold all validation results
                    validation_results = dict()

                    for val_set in validation_sets_onset:
                        # Validate the model checkpoint on each validation dataset
                        validation_results[val_set.name()] = evaluate_onset(model=model,
                                                                    eval_set=val_set,
                                                                    multipliers=multipliers,
                                                                    THRESHOLD=0.5,
                                                                    writer=writer,
                                                                    i=batch_count,
                                                                    device=device,
                                                                    eq_fn=eq_fn,
                                                                    eq_kwargs=eq_kwargs,
                                                                    gm_kwargs=gm_kwargs)

                    # Make sure model is on correct device and switch to training mode
                    model = model.to(device)
                    model.train()

                    if cudnn_benchmarking:
                        # Re-enable benchmarking after validation
                        torch.backends.cudnn.benchmark = True

                        if decay_scheduler_onset.patience and not warmup_scheduler_onset.is_active() and i >= n_epochs_late_start:
                            # Step the learning rate decay scheduler by logging the validation metric for the checkpoint
                            decay_scheduler_onset.step(validation_results[validation_criteria_set][validation_criteria_metric])

                        # Extract the result on the specified metric from the validation results for comparison
                        current_score = validation_results[validation_criteria_set][validation_criteria_metric]

                        if best_results is not None:
                            # Extract the currently tracked best result on the specified metric for comparison
                            best_score = best_results[validation_criteria_set][validation_criteria_metric]

                        if best_results is None or \
                                (validation_criteria_maximize and current_score > best_score) or \
                                (not validation_criteria_maximize and current_score < best_score):
                            print(f'New best at {batch_count} iterations...')

                            # Set current checkpoint as best
                            best_model_checkpoint = batch_count
                            # Update best results
                            best_results = validation_results
                            # Reset number of checkpoints
                            n_checkpoints_elapsed = 0
                        else:
                            # Increment number of checkpoints
                            n_checkpoints_elapsed += 1

                        if n_checkpoints_early_stop is not None and n_checkpoints_elapsed >= n_checkpoints_early_stop:
                            # Early stop criteria has been reached
                            early_stop_criteria = True

                            break

                        if early_stop_criteria:
                            # Stop training
                            break
        else:
            print('No Phase3 training...')

    print(f'Achieved best results at {best_model_checkpoint} iterations...')
# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_drill.datasets import NoteDataset
from timbre_drill.framework.objectives import *
from timbre_drill.utils import *

# Regular imports
import librosa
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def evaluate(model, eval_set, multipliers, THRESHOLD=0.5, writer=None,
             i=0, device='cpu', eq_fn=None, eq_kwargs={}, gm_kwargs={},
             plotNsave=False):
    # Initialize a new evaluator for the dataset
    evaluator = MultipitchEvaluator()

    # Add model to selected device and switch to evaluation mode
    model = model.to(device)
    model.eval()

    plot_path_root = './generated/experiments/test_plot'

    precision = 0
    recall = 0
    accuracy = 0
    f1_score = 0

    with torch.no_grad():
        # Loop through tracks
        for j, data in enumerate(eval_set):
            # Determine which track is being processed
            track = data[constants.KEY_TRACK]
            # Extract audio and add to the appropriate device
            audio = data[constants.KEY_AUDIO].to(device).unsqueeze(0)
            # Extract ground-truth targets as a Tensor
            ground_truth = torch.Tensor(data[constants.KEY_GROUND_TRUTH])

            if isinstance(eval_set, NoteDataset):
                # Extract frame times of ground-truth targets as reference
                times_ref = data[constants.KEY_TIMES]
                # Obtain the ground-truth note annotations
                pitches, intervals = eval_set.get_ground_truth(track)
                # Convert note pitches to Hertz
                pitches = librosa.midi_to_hz(pitches)
                # Convert the note annotations to multi-pitch annotations
                multi_pitch_ref = eval_set.notes_to_multi_pitch(pitches, intervals, times_ref)
            else:
                # Obtain the ground-truth multi-pitch annotations
                times_ref, multi_pitch_ref = eval_set.get_ground_truth(track)

            # Compute full set of spectral features
            features = model.get_all_features(audio, onset_mode=True)

            # Extract relevant feature sets
            input = features['hcqt'] # (B, 2, F(per), T)
            onset_selection = features['onset_selection'] # (B, F(one), T)
            weak_label_neg = features['pitch_negative_label'] # (B, F(per), T)
            weak_label_pos = features['pitch_positive_label'] # (B, F(per), T)

            # Process features to obtain logits
            output = model(input)

            pitch_logits = output['pitch_logits']
            pitch_const = output['pitch_const']
            pitch_salience = output['pitch_salience']

            # Determine the times associated with predictions
            times_est = model.sliCQ.get_times(model.sliCQ.get_expected_frames(audio.size(-1)))
            # Perform peak-picking and thresholding on the activations
            activations = threshold(filter_non_peaks(to_array(pitch_salience)), THRESHOLD).squeeze(0)

            # Convert the activations to frame-level multi-pitch estimates
            multi_pitch_est = eval_set.activations_to_multi_pitch(activations, model.sliCQ.get_midi_freqs())

            # Compute results for this track using mir_eval multi-pitch metrics
            results = evaluator.evaluate(times_est, multi_pitch_est, times_ref, multi_pitch_ref)

            # precision += results['mpe/precision']
            # recall += results['mpe/recall']
            # accuracy += results['mpe/accuracy']
            # f1_score += results['mpe/f1-score']

            # print(results)

            # Store the computed results
            evaluator.append_results(results)

            # PITCH BINARY CROSS ENTROPY
            pitch_support_loss = compute_support_loss(pitch_logits, weak_label_neg)
            pitch_harmonic_loss = compute_harmonic_loss(pitch_logits, weak_label_pos)

            # Compute sparsity loss for the track
            pitch_sparsity_loss = compute_sparsity_loss(pitch_salience)
            # Compute supervised BCE loss for the batch
            #pitch_supervised_loss = compute_supervised_loss(pitch_salience, ground_truth.to(device).unsqueeze(0))

            pitch_time_sim_loss = compute_time_sim_loss(pitch_logits)

            # Compute the total loss for the track
            total_loss = multipliers['support_p'] * pitch_support_loss + \
                         multipliers['harmonic_p'] * pitch_harmonic_loss + \
                         multipliers['time_sim'] * pitch_time_sim_loss + \
                         multipliers['sparsity_p'] * pitch_sparsity_loss #+ \
                         #multipliers['supervised'] * pitch_supervised_loss

            if eq_fn is not None:
                # Compute timbre loss for the track using specified equalization
                pitch_timbre_loss = compute_pitch_timbre_loss(model, input, pitch_logits, eq_fn, **eq_kwargs)
                # Store the timbre loss for the track
                evaluator.append_results({'loss/timbre' : pitch_timbre_loss.item()})
                # Add the timbre loss to the total loss
                total_loss += multipliers['timbre_p'] * pitch_timbre_loss

            # Compute geometric loss for the track
            pitch_geometric_loss = compute_pitch_geometric_loss(model, input, pitch_logits, **gm_kwargs)
            # Store the geometric loss for the track
            evaluator.append_results({'loss/geometric' : pitch_geometric_loss.item()})
            # Add the geometric loss to the total loss
            total_loss += multipliers['geometric_p'] * pitch_geometric_loss

            # for key_loss, val_loss in losses.items():
            #     # Store the model loss for the track
            #     evaluator.append_results({f'loss/{key_loss}' : val_loss.item()})
            #     # Add the model loss to the total loss
            #     total_loss += multipliers.get(key_loss, 1) * val_loss

            # Store all losses for the track
            evaluator.append_results({'loss/pitch_support' : pitch_support_loss.item(),
                                      'loss/pithc_harmonic' : pitch_harmonic_loss.item(),
                                      'loss/pitch_sparsity' : pitch_sparsity_loss.item(),
                                      #'loss/pitch_supervised' : pitch_supervised_loss.item(),
                                      'loss/total' : total_loss.item()})
            
            if 'keyboard_electronic_001-067-075' in track or j == 0:
                plot_ground_truth = ground_truth
                plot_pitch_salience = pitch_salience
                plot_pitch_logits = pitch_logits
                #plot_onset_salience = onset_salience
                plot_weak_label_pos = weak_label_pos
                plot_weak_label_neg = weak_label_neg
                plot_pitch_const = pitch_const
                #plot_pitch_trans = pitch_trans
                #plot_onset_logits = onset_logits
                #plot_onset_selection = onset_selection

            if plotNsave:
                fig, axs = plt.subplots(2, 1, figsize=(48, 32))
                axs[0].imshow(pitch_salience[0].cpu(), cmap='gray')
                axs[0].set_title('pitch salience')
                axs[1].imshow(ground_truth.cpu(), cmap='gray')
                axs[1].set_title('groundtruth')
                plt.tight_layout()
                plot_path = plot_path_root + '/' + track
                plt.savefig(plot_path)

            break

        # Compute the average for all scores
        average_results, _ = evaluator.average_results()

        print(eval_set.name())
        print(average_results)

        if writer is not None:
            # Loop through all computed scores
            for key in average_results.keys():
                # Log the average score for this dataset
                writer.add_scalar(f'{eval_set.name()}/{key}', average_results[key], i)

            # Add channel dimension to input/outputs
            ground_truth = plot_ground_truth.unsqueeze(-3)
            pitch_salience = plot_pitch_salience.unsqueeze(-3)
            pitch_logits = plot_pitch_logits.unsqueeze(-3)
            #onset_salience = plot_onset_salience.unsqueeze(-3)
            weak_label_pos = plot_weak_label_pos.unsqueeze(-3)
            weak_label_neg = plot_weak_label_neg.unsqueeze(-3)
            pitch_const = plot_pitch_const.unsqueeze(-3)
            #pitch_trans = plot_pitch_trans.unsqueeze(-3)
            # onset_logits = plot_onset_logits.unsqueeze(-3)
            # onset_selection = plot_onset_selection.unsqueeze(-3)

            # Remove batch dimension from inputs
            pitch_salience = pitch_salience.squeeze(0)
            pitch_logits = pitch_logits.squeeze(0)
            #onset_salience = onset_salience.squeeze(0)
            weak_label_pos = weak_label_pos.squeeze(0)
            weak_label_neg = weak_label_neg.squeeze(0)
            pitch_const = pitch_const.squeeze(0)
            #pitch_trans = pitch_trans.squeeze(0)
            # onset_logits = onset_logits.squeeze(0)
            # onset_selection = onset_selection.squeeze(0)

            # Visualize predictions for the final sample of the evaluation dataset

            writer.add_image(f'{eval_set.name()}/ground-truth', ground_truth.flip(-2), i)

            writer.add_image(f'{eval_set.name()}/transcription', pitch_salience.flip(-2), i)
            writer.add_image(f'{eval_set.name()}/pitch logits', pitch_logits.flip(-2), i)

            # writer.add_image(f'{eval_set.name()}/onset logits', onset_logits.flip(-2), i)
            # writer.add_image(f'{eval_set.name()}/onset detection', onset_salience.flip(-2), i)

            writer.add_image(f'{eval_set.name()}/reconst', pitch_const.flip(-2), i)

            #writer.add_image(f'{eval_set.name()}/pre-trans(real)', pitch_trans.flip(-2), i)

            writer.add_image(f'{eval_set.name()}/Weak Label_h (dB)', weak_label_pos.flip(-2), i)
            writer.add_image(f'{eval_set.name()}/Weak Label_1 (dB)', weak_label_neg.flip(-2), i)
            #writer.add_image(f'{eval_set.name()}/Spectral Flux', onset_selection.flip(-2), i)


    return average_results

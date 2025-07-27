# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_drill.datasets import NoteDataset
from timbre_drill.framework.objectives import *
from timbre_drill.utils import *
from timbre_drill.framework import CFP_HMLC, Note_level_processing, note_transcription
from timbre_drill.framework.hmlc_cfp import *

# Regular imports
import librosa
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import mir_eval


def predict(model, eval_set, multipliers, THRESHOLD=0.5, THRESHOLD_O=0.5, writer=None,
             i=0, device='cpu', eq_fn=None, eq_kwargs={}, gm_kwargs={}, midi_path=None,
             plotNsave=False):
    # Initialize a new evaluator for the dataset
    evaluator = MultipitchEvaluator()

    # CFP_HMLC note level process
    SF_maxfilt_bw = 61 # bandwidth of the maximal filter in computing onsets in bins
    SF_thres = 0.1 # threshold of onset detection
    lam = 1E-3 # parameter of the DP-based pitch contour tracking
    Note_th = 5 # threshold of note length, in frames
    Repeated_note_th = 15 # threshold of repeated note segmentation, in frames
    Transition_th = 3 # threshold of note transition
    note_level = Note_level_processing(SF_maxfilt_bw, SF_thres, lam, Note_th, Repeated_note_th, Transition_th)

    notetrans = note_transcription(model.CFP_HMLC, note_level)

    # Add model to selected device and switch to evaluation mode
    model = model.to(device)
    model.eval()

    plot_path_root = './generated/experiments/test_plot'

    Precision = 0
    Recall = 0
    F_measure = 0
    Average_overlap_ratio = 0
    F_measure_no_off = 0
    Onset_F = 0
    Offset_F = 0

    with torch.no_grad():
        # Loop through tracks
        for j, data in enumerate(eval_set):

            # if j!=1:
            #     continue
            
            Note_all = {'coarse': [], 'fine': []} #[]

            # Determine which track is being processed
            track = data[constants.KEY_TRACK]
            # Extract audio and add to the appropriate device
            audio = data[constants.KEY_AUDIO].to(device).unsqueeze(0)
            # Extract ground-truth targets as a Tensor
            ground_truth = torch.Tensor(data[constants.KEY_GROUND_TRUTH])

            midi_table = model.sliCQ.get_midi_freqs()

            if isinstance(eval_set, NoteDataset):
                # Extract frame times of ground-truth targets as reference
                times_ref = data[constants.KEY_TIMES]
                # Obtain the ground-truth note annotations
                pitches, intervals = eval_set.get_ground_truth(track)
                # Convert note pitches to Hertz
                pitches = librosa.midi_to_hz(pitches)
                # Convert the note annotations to multi-pitch annotations
                multi_pitch_ref = eval_set.notes_to_multi_pitch(pitches, intervals, times_ref)

                intervals_ref = intervals
                note_pitches_ref = np.expand_dims(pitches, axis=1)
            else:
                # Obtain the ground-truth multi-pitch annotations
                times_ref, multi_pitch_ref = eval_set.get_ground_truth(track)
                intervals_ref, note_pitches_ref = eval_set.get_note_ground_truth(track)

            # print(intervals_ref)
            # print(note_pitches_ref)
            # Compute full set of spectral features
            features = model.get_all_features(audio, onset_mode=True)

            # Extract relevant feature sets
            input = features['hcqt'] # (B, 2, F(per), T)
            onset_select = features['onset_selection'] # (B, F(one), T)
            #weak_label_neg = features['pitch_negative_label'] # (B, F(per), T)
            #weak_label_pos = features['pitch_positive_label'] # (B, F(per), T)

            # Process features to obtain logits
            #pitch_trans, pitch_const, latents, pitch_logits, pitch_salience, note_logits, note_salience, onset_logits, onset_salience, losses = model(input, transcribe=True)
            
            #pitch_salience, onset_salience = model.inference(input)
            pitch_salience, onset_salience = model.inference_sep(input)
            
            #note = pitch_contour_reduce(pitch_salience, bins_per_semitone=model.cqt_params['bins_per_octave']//12)

            # Determine the times associated with predictions
            times_est = model.sliCQ.get_times(model.sliCQ.get_expected_frames(audio.size(-1)))

            # Perform peak-picking and thresholding on the activations
            activations = threshold(filter_non_peaks(to_array(pitch_salience)), THRESHOLD)
            # pitch_salience_to_dp = to_array(pitch_salience).squeeze(0)

            # Convert the activations to frame-level multi-pitch estimates
            multi_pitch_est = eval_set.activations_to_multi_pitch(activations, model.sliCQ.get_midi_freqs())

            results = evaluator.evaluate(times_est, multi_pitch_est, times_ref, multi_pitch_ref)

            evaluator.append_results(results)

            print(track)
            print(results)

            ############################################################################
            ########                        CFP-HMLC                          ##########
            ############################################################################
            
            # # onset time
            # to_find_peak = torch.sum(onset_salience.squeeze(0), dim=0).cpu().detach().numpy()

            # _, onset_time = findpeaks_time(to_find_peak, SF_thres)
            # onset_time = onset_time[0] # time axis index
            
            # note_locs = note_level.note_seg_dp(pitch_salience_to_dp, onset_time)

            # Note = note_level.post_proc(note_locs[0, :], activations, onset_time)
            
            # Note_all['coarse'].extend(Note['coarse'])
            # Note_all['fine'].extend(Note['fine'])

            # # Note_all['coarse'] = remove_outlier(Note_all['coarse'])
            # # Note_all['fine'] = remove_outlier(Note_all['fine'])

            # note_to_midi(Note_all['coarse'], midi_path, 0)

            # # Create Note event
            # Note = Note_all['coarse']

            # intervals_est, note_pitches_est = None, None

            # NumOfNote = len(Note)

            # for k in range(NumOfNote):
            #     # Retrieve the MIDI note number for this note name
            #     note_number = int(np.round(21+Note[k]['pitch'][0,0]/5.0)) #int(np.round(48+Note[j]['pitch'][0,0]/5.0))#int(12.0*np.log2(note_freq/440.0)+69.0)
            #     note_hz = np.array(librosa.midi_to_hz(note_number)).astype('float')
            #     # note_number = raga_list[np.where(abs(note_number-raga_list)==np.amin(abs(note_number-raga_list)))][0]
            #     # 1.16099773e-02
            #     start_time = np.array(Note[k]['time'][0,0]*0.01161).astype('float')
            #     end_time = np.array(Note[k]['time'][0,-1]*0.01161).astype('float')

            #     _pitches = np.expand_dims(note_hz, axis=(0, 1))
            #     _onset_time = np.expand_dims(start_time, axis=(0, 1))
            #     _offset_time = np. expand_dims(end_time, axis=(0, 1))

            #     _interval = np.concatenate((_onset_time, _offset_time)).T

            #     if intervals_est is None or note_pitches_est is None:
            #         intervals_est, note_pitches_est = _interval, _pitches
            #     else:
            #         intervals_est = np.append(intervals_est, _interval, axis=0)
            #         note_pitches_est = np.append(note_pitches_est, _pitches, axis=0)

            ############################################################################
            ########                        Basic-Pitch                       ##########
            ############################################################################
            

            estimated_notes = output_to_notes_polyphonic(
                to_array(pitch_salience).T,
                to_array(onset_salience).T,
                onset_thresh=THRESHOLD_O,
                frame_thresh=THRESHOLD,
                infer_onsets=False,
                min_note_len=5,
                min_freq=None,
                max_freq=None,
                melodia_trick=True,
            )

            intervals_est = [
                    (times_est[note[0]], times_est[note[1]]) for note in estimated_notes
            ]

            note_pitches_est = [
                    librosa.midi_to_hz((note[2]-21)//5+21) for note in estimated_notes
            ]

            intervals_ref = np.array(intervals_ref)
            note_pitches_ref = np.array(note_pitches_ref).squeeze(1)
            intervals_est = np.array(intervals_est)
            note_pitches_est = np.array(note_pitches_est)

            # print(intervals_ref.shape)
            # print(note_pitches_ref.shape)
            # print(intervals_est.shape)
            # print(note_pitches_est.shape)

            score = mir_eval.transcription.evaluate(intervals_ref, note_pitches_ref, intervals_est, note_pitches_est)

            print(score)

            Precision += score['Precision']
            Recall += score['Recall']
            F_measure += score['F-measure']
            F_measure_no_off += score['F-measure_no_offset']
            Average_overlap_ratio += score['Average_Overlap_Ratio']
            Onset_F += score['Onset_F-measure']
            Offset_F += score['Offset_F-measure']

            # if j==1:
            #     fig, axs = plt.subplots(3, 1, figsize=(12, 8))

            #     axs[0].imshow(pitch_salience[:,:1000].cpu(), cmap='gray')
            #     axs[0].set_title('pitch')
            #     axs[1].imshow(onset_salience[:,:1000].cpu(), cmap='gray')
            #     axs[1].set_title('onset')
            #     axs[2].imshow(ground_truth[:,:1000], cmap='gray')  
            #     axs[2].set_title('Groundtruth')

            #     plt.tight_layout()

            #     plt.show()
            #     assert(False)

    average_results, _ = evaluator.average_results()

    print('Result of dataset: ' ,eval_set.name())
    print(average_results)

    Precision /= len(eval_set)
    Recall /= len(eval_set)
    F_measure /= len(eval_set)
    F_measure_no_off /= len(eval_set)
    Average_overlap_ratio /= len(eval_set)
    Onset_F /= len(eval_set)
    Offset_F /= len(eval_set)

    print("Avg. result")
    print("Precision: ", Precision)
    print("Recall: ", Recall)
    print("F_measure: ", F_measure)
    print("F_measure_no_offset: ", F_measure_no_off)
    print("Average_overlap_ratio: ", Average_overlap_ratio)
    print("Onset_F: ", Onset_F)
    print("Offset_F: ", Offset_F)

    return None

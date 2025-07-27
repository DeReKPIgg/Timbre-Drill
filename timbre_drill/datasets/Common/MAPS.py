from timbre_drill.utils.data import *
from .. import AMTDataset

import pandas as pd
import numpy as np
import os


class MAPS(AMTDataset):
    """
    Implements the top-level wrapper for the MusicNet dataset (https://zenodo.org/record/5120004).
    """

    @staticmethod
    def available_splits():
        """
        Obtain a list of available (pre-defined) dataset splits.

        Returns
        ----------
        splits : list of strings
          Names of originally proposed splits
        """

        #splits = ['MAPS_ENSTDkAm_2', 'MAPS_ENSTDkCl_2'] # test 

        # train
        splits = ['MAPS_AkPnBcht_1', 'MAPS_AkPnBcht_2', 
                  'MAPS_AkPnBsdf_1', 'MAPS_AkPnBsdf_2',
                  'MAPS_AkPnCGdD_1', 'MAPS_AkPnCGdD_2',
                  'MAPS_AkPnStgb_1', 'MAPS_AkPnStgb_2',
                  'MAPS_ENSTDkAm_2', 'MAPS_ENSTDkCl_2',
                  'MAPS_SptkBGAm_1', 'MAPS_SptkBGAm_2',
                  'MAPS_SptkBGCl_1', 'MAPS_SptkBGCl_2',
                  'MAPS_StbgTGd2_1', 'MAPS_StbgTGd2_2']

        return splits

    def get_tracks(self, split):
        """
        Get the names of the tracks in the dataset.

        Parameters
        ----------
        split : string
          String indicating train or test split

        Returns
        ----------
        tracks : list of strings
          Names of tracks belonging to the split
        """
        tracks = []

        '''Read Txt file'''
        # print(split)
        # tmp_file = os.path.join(self.base_dir, split + '.txt')
        # file = open(tmp_file, "r")
        # data = file.read()
        # datas = data.split("\n")
        # file.close()
        # for data in datas: # thouugh data is already plural
        #   tmp_url = os.path.join(self.base_dir, split, split.split('_')[1], 'MUS', data)
        #   tracks.append(tmp_url)

        '''Get whole directory'''
        dir_path = os.path.join(self.base_dir, split, split.split('_')[1], 'MUS')
        tracks = [d for d in os.listdir(dir_path) if d.endswith('.wav')]
        tracks = [os.path.join(split, t) for t in tracks]

        return tracks

    def get_audio_path(self, track):
        
        '''Read Txt file'''
        #return track

        '''Get whole directory'''
        split, name = os.path.split(track)
        wav_path = os.path.join(self.base_dir, split, split.split('_')[1], 'MUS', name)

        return wav_path

    def get_ground_truth_path(self, track):
        audio_path = self.get_audio_path(track)
        txt_path = audio_path.replace('.wav', '.txt')

        return txt_path

    def get_ground_truth(self, track):
        """
        Extract the ground-truth for the specified track.

        Parameters
        ----------
        track : string
          MusicNet track name

        Returns
        ----------
        pitches : ndarray (L)
          Array of note pitches
        intervals : ndarray (L x 2)
          Array of corresponding onset-offset time pairs
        """
        csv_path = self.get_ground_truth_path(track)

        # Load tabulated note data from the csv file
        note_entries = pd.read_csv(csv_path, sep="\t").to_numpy()

        # Unpack the relevant note attributes and convert them to integers
        onsets, offsets, pitches = note_entries[:, (0, 1, 2)].T
        pitches = pitches.astype(int)
        # Construct intervals for the notes and convert to seconds
        intervals = np.concatenate(([onsets], [offsets])).T

        return pitches, intervals

    @classmethod
    def download(cls, save_dir):
        """
        Download the MusicNet dataset to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the contents of MusicNet
        """

        # Create top-level directory
        super().download(save_dir)

        # URL pointing to the tar file containing audio/annotations
        anno_url = 'https://zenodo.org/record/5120004/files/musicnet.tar.gz'

        # Construct a path for saving the annotations
        anno_path = os.path.join(save_dir, os.path.basename(anno_url))

        # Download the tar file containing annotations
        stream_url_resource(anno_url, anno_path, 1000 * 1024)

        # Untar the downloaded file and remove it
        unzip_and_remove(anno_path, tar=True)

        # Move contents of untarred directory to the base directory
        change_base_dir(save_dir, os.path.join(save_dir, 'musicnet'))

        # URL pointing to the tar file containing MIDI for all tracks
        midi_url = 'https://zenodo.org/record/5120004/files/musicnet_midis.tar.gz'

        # Construct a path for saving the MIDI files
        midi_path = os.path.join(save_dir, os.path.basename(midi_url))

        # Download the tar file containing MIDI files
        stream_url_resource(midi_url, midi_path, 1000 * 1024)

        # Untar the downloaded file and remove it
        unzip_and_remove(midi_path, tar=True)

        # URL pointing to the metadata file for all tracks
        meta_url = 'https://zenodo.org/record/5120004/files/musicnet_metadata.csv'

        # Construct a path for saving the metadata
        meta_path = os.path.join(save_dir, os.path.basename(meta_url))

        # Download the metadata file
        stream_url_resource(meta_url, meta_path, 1000 * 1024)

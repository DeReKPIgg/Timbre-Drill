import torch

__all__ = [
    'pitch_contour_reduce',
]

def pitch_contour_reduce(pitch_contour, bins_per_semitone=5, threshold=0.5):
    #(B x F(per) x T)
    note = torch.reshape(pitch_contour, (pitch_contour.size(0), bins_per_semitone, -1, pitch_contour.size(-1)))
    note = torch.mean(note, dim=1)

    return note

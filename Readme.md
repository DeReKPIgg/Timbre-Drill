Source code of "SELF-SUPERVISED NOTE TRACKING AND MULTI-PITCH ESTIMATION VIA RECONSTRUCTION-BASED LEARNING" by Heng-Hsiu Hu and Li Su.

###  ***  IMPORTANT CORRECTION !!  *** <br />
The URMP training set and testing set have overlapping problem, we're now checking on all experiments in Table 1.

## Enviroment Setup
```
conda create ss-nt-mpe-rc python=3.10.12
conda activate ss-nt-mpe-rc

pip install -r requirements.txt

conda install "setuptools<60.0.0"
pip install torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

## Training
```
python train.py
```

## Testing
```
python test.py
```

## Acknowledgement
The code is based on [SS-MPE](https://github.com/cwitkowitz/ss-mpe), [Timbre-Trap](https://github.com/sony/timbre-trap) and [Basic-Pitch](https://github.com/spotify/basic-pitch).
Thanks for their awesome works.

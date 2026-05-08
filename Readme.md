Source code of "SELF-SUPERVISED NOTE TRACKING AND MULTI-PITCH ESTIMATION VIA RECONSTRUCTION-BASED LEARNING" by Heng-Hsiu Hu and Li Su.

## Enviroment Setup
```
conda create ss-nt-mpe-rc python=3.10
conda activate ss-nt-mpe-rc

pip install "setuptools<70.0.0" wheel
pip install crepe==0.0.15 --no-build-isolation
pip install -r requirement.txt
```

TODO: refactor functions 0o0

## Acknowledgement
The code is based on [SS-MPE](https://github.com/cwitkowitz/ss-mpe), [Timbre-Trap](https://github.com/sony/timbre-trap) and [Basic-Pitch](https://github.com/spotify/basic-pitch).
Thanks for their awesome works.

# Pre-trained Models

This folder contains pre-trained models used in the project.

The Persian FastText model `cc.fa.300.bin` is not included in this repository because of its large file size.

To download the model, run:

```python
import fasttext.util

fasttext.util.download_model('fa', if_exists='ignore')
```

After downloading, place the file in this folder:

```
models/
└── cc.fa.300.bin
```
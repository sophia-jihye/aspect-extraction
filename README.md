## Repositories to be explored
* [jeetp465; Aspect extraction for opinion mining with a deep convolutional neural network (2016)](https://github.com/jeetp465/Aspect-Based-Sentiment-Analysis)
* [howardhsu; Double Embeddings and CNN based Sequence Labeling for Aspect Extraction (2018)](https://github.com/howardhsu/DE-CNN)
* [yafangy; Double Embeddings and CNN-based Sequence Labeling for Aspect Extraction (2018)](https://github.com/yafangy/Review_aspect_extraction)
> In my "triple embeddings" model, feature map output from the conv layer is concatenated with domain embeddings of the word as well as its part-of-speech tagger (stanford), and then it is feeded to a fully-connected neural network with one hidden layer of size 50.
    
# Aspect extraction from product reviews with Tensorflow
This repo has multiple sequential models for aspect extraction from product reviews.

## Citation
Poria, S., Cambria, E. and Gelbukh, A., 2016. Aspect extraction for opinion mining with a deep convolutional neural network. Knowledge-Based Systems, 108, pp.42-49.

## Task
Given a sentence, the task is to extract aspects. Here is an example

```
I like the battery life of this phone"

Converting this sentence to IOB would look like this -

I O
like O
the O
battery B-A
life I-A
of O
this O
phone O
```


## Model

Similar to [Ma and Hovy](https://arxiv.org/pdf/1603.01354.pdf).
- concatenate final states of a bi-lstm on character embeddings to get a character-based representation of each word
- concatenate this representation to a standard word vector representation (GloVe here)
- run a bi-lstm on each sentence to extract contextual representation of each word
- decode with a linear chain CRF

Similar to [Collobert et al.] (http://ronan.collobert.com/pub/matos/2011_nlp_jmlr.pdf)
- form a window around the word to tag
- apply MLP on that window
- obtain logits
- apply viterbi (CRF) for sequence tagging

Similar to [Poria et al.](https://www.sciencedirect.com/science/article/pii/S0950705116301721)
- form a window around the word to tag
- apply CNN on that window
- apply maxpool on that window (Caution: different from global maxpool)
- obtain logits
- apply CRF for sequence tagging

## Details

Download Glove embeddings (GloVe: http://nlp.stanford.edu/data/glove.840B.300d.zip )

1. [DO NOT MISS THIS STEP] Build vocab from the data and extract trimmed glove vectors according to the config in `model/config.py`.

```
python build_data.py
```

2. Train the model with

```
python train.py
```


3. Evaluate and interact with the model with
```
python evaluate.py
```


Data iterators and utils are in `model/data_utils.py` and the model with training/test procedures is in `model/aspect_model.py`


## Training Data

* `B-A` means that it starts a new phrase
* `I-A` means that the word is inside a phrase
* O: character O

```
The	O
duck	B-A
confit	I-A
is	O
always	O
amazing	O
and	O
the	O
foie	B-A
gras	I-A
terrine	I-A
with	I-A
figs	I-A
was	O
out	O
of	O
this	O
world	O

The	O
wine	B-A
list	I-A
is	O
interesting	O
and	O
has	O
many	O
good	O
values	O
```

Once you have produced your data files, change the parameters in `config.py` like

```
# dataset
filename_train = "data/ABSA16_Restaurants_Train_SB1_v2_mod.iob"
filename_dev = "data/EN_REST_SB1_TEST_2016_mod.iob"
filename_test = "data/EN_REST_SB1_TEST_2016_mod.iob"
```

## Result

Chunk based evaluation

```
Laptop 2014 -> F1 - 79.93

Restaurant 2014 -> F1 - 84.22
```
Partial matching based evaluation
```
Laptop 2014 -> F1 - 86.84
Restaurant 2014 -> F1 - 88.42
```

## License

This project is licensed under the terms of the apache 2.0 license (as Tensorflow and derivatives). If used for research, citation would be appreciated.


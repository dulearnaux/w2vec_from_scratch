# Word2Vec from Scratch, using `numpy`

This repo implements word2vec from scratch, using `numpy`. It does so as described in the [original 2013 word2vec paper](original_papers/Efficient%20Estimation%20of%20Word%20Representations%20in%0AVector%20Space.pdf).

Both Continuous bag of words and s-gram approaches are implemented. And the negative sampling described in the [followup paper](original_papers/Distributed%20Representations%20of%20Words%20and%20Phrases%20(more%20effecient%20follow%20up).pdf) is also implemented for each approach.

If you actually want to use word2vec for yourself, you should probably use a package like Gensim. The purpose of this repo is simply to demonstrate how word2vec works, by using only `numpy` (although `nltk` was used for some basic tokenization tasks).


### Data Description

The data used was 4.1 GB of english language news stories stored in 100 files.
 - [description: https://www.statmt.org/lm-benchmark/](https://www.statmt.org/lm-benchmark/)
 - [download](https://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz)
 - The data was trained on data in the directory: `/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled` 
 
This was processed by `clean_raw_data.py`:
 - punctuation removed
 - tokenised (into whole english words).
 - stop words removed (`nltk.corpus.stopwords`)
 - corpus was down sampled according to Mikolov et. al. 2013, Section 2.3
 - minimum frequency of words was set to 1000. Low frequency words were discarded from the corpus.
 - News stories that were less than 10 words long (after the above processing) were removed. 

Went from 4.1GB to 2.8GB of data. Final vocab is 21,927 words.


### Run training of models.

As the models trains,
* the model will get saved periodically (every 10 batches)
* the loss curve plotted to a file. Monitor the plot to determine when sufficient training is completed.
* at each epoch a checkpoint will be saved. 

#### Continuous Bag of Words
To train the **Continuous Bag of Words** model run the `train_numpy_model.py`
file as follows:

```shell
python train_numpy_model.py \
    --model_file=cbow.model \  # Save model here. Relative to python file.
    --data_file=data/toy_data_41mb_raw.txt.train \ 
    --overwrite_model \  # start new model, or overwrite old one.
    --type=cbow \  # Use CBOW type model.
    --window=7 \
    --vector_dim=300 \
    --batch_size=512 \
    --epochs=1000 \
    --alpha=0.005
```

#### Continuous Bag of Words with Negative Samples
To train the **Continuous Bag of Words with Negative Samples** model run the
`train_numpy_model.py` file as follows:

```shell
python train_numpy_model.py \
    --model_file=cbow_neg.model \  # Save model here. Relative to python file.
    --data_file=data/toy_data_41mb_raw.txt.train \ 
    --overwrite_model \  # start new model, or overwrite old one.
    --type=cbow_neg \  # Use CBOW type model.
    --window=7 \
    --neg_samples=6 \
    --vector_dim=300 \
    --batch_size=512 \
    --epochs=1000 \
    --alpha=0.1
```


#### Skip-gram
To train the **Skip-gram** model run the `train_numpy_model.py`
file as follows:

```bash
python train_numpy_model.py \
    --model_file=sgram.model \  # Save model here. Relative to python file.
    --data_file=data/toy_data_41mb_raw.txt.train \
    --overwrite_model \  # start new model, or overwrite old one.
    --type=sgram \  # Use S-gram type model.
    --window=5 \
    --vector_dim=300 \
    --batch_size=512 \
    --alpha=0.01
```

#### Skip-gram with Negative Samples
To train the **Skip-gram** model run the `train_numpy_model.py`
file as follows:

```bash
python train_numpy_model.py \
    --model_file=sgram_neg.model `# Save model here. Relative to python file.` \
    --data_file=data/*.train[0-9][0-9] \
    --overwrite_model `# start new model, or overwrite old one.` \
    --type=sgram_neg `# Use S-gram type model.` \
    --window=5 \
    --neg_samples=6 \
    --vector_dim=300 \
    --batch_size=512 \
    --alpha=0.01
```

## Some of Math for Negative Sampling

### Negative Sampling Loss Function

```math
L = -log\left(\sigma\left(c_{pos}.h\right)\right) - \sum log\left(\sigma\left(-c_{neg}.h\right)\right)
```
  
Where:  
- **h** = hidden projection layer. I.e. the column from the input matrix $W_1$ corresponding to the input word. In the case of multiple input words (e.g. for CBOW), **h** is the average of the multiple columns.  
- $c_{pos}$ = Row from $W_2$ corresponding to positive words. For CBOW, there is only a single positive word. For S-gram, there are window-1 positive words.  
- $c_{neg}$ = Similar to $c_{pos}$, but corresponding to negative words. There are **k** negative words. **k** is a hyperparamter of negative sampling. 2-5 is recommended.  
- $\sigma$ is the sigmoid function: $\frac{1}{1+e^{-x}}$
- $c_{pos}.h$ and $c_{neg}.h$ are the inner dot product. I.e. a scalar.
  
Note:
  Since $\sigma(x) = 1 - \sigma(-x)$, The loss function can be represented as below. Which perhaps more inutitively suggests we want to maximize the positive word dot products with the input word projection, **$$h$$**, from **$$W_1$$**. And minimize the same for the negative words.
```math
L = -log\left(\sigma\left(c_{pos}.h\right)\right) + \sum log\left(\sigma\left(c_{neg}.h\right)\right) - 1  
```
  
  
### Negative Sampling Gradients

- For the input matrix $W_1$, only the input words colum is changed, which is represented by **h**. So we want:
```math
\frac{\partial L}{\partial h}
```
- For the output matrix $W_2$, only the rows for the context words and negative sample words change. So we want:
```math
\frac{\partial L}{\partial c_{pos}} and \frac{\partial L}{\partial c_{neg}}
```

#### W.R.T $W_1$, h.
  
Break out $\frac{\partial L}{\partial h}$ into the two parts for pos and neg words.

##### Positive words
$` \frac{\partial L_{pos}}{\partial h} = \frac{\partial}{\partial h} \left(-log\left(\sigma\left(c_{pos}.h\right)\right)\right) `$  
$` \frac{\partial L_{pos}}{\partial h} = \frac{-1}{\sigma(c_{pos}.h)}.\frac{\partial}{\partial h}\sigma(c_{pos}.h) `$  
$` \frac{\partial L_{pos}}{\partial h} = \frac{-1}{\sigma(c_{pos}.h)}.\left(\sigma(c_{pos}.h)(1-\sigma(c_{pos}.h))\right)\frac{\partial}{\partial h}(c_{pos}.h) `$  
$` \frac{\partial L_{pos}}{\partial h} = -1 .\left(1-\sigma(c_{pos}.h)\right).c_{pos} `$  
$` \frac{\partial L_{pos}}{\partial h} = \left(\sigma(c_{pos}.h) - 1\right).c_{pos} `$  

##### Negative words
$` \frac{\partial L_{neg}}{\partial h} = \frac{\partial}{\partial h} \sum\left(-log\left(\sigma\left(-c_{neg}.h\right)\right)\right) `$  
$` \frac{\partial L_{neg}}{\partial h} = \sum \frac{-1}{\sigma(-c_{neg}.h)}. \frac{\partial}{\partial h}\sigma(-c_{neg}.h) `$  
$` \frac{\partial L_{neg}}{\partial h} = \sum \frac{-1}{\sigma(-c_{neg}.h)}.\left(\sigma(-c_{neg}.h)(1-\sigma(-c_{neg}.h))\right)\frac{\partial}{\partial h}(-c_{neg}.h) `$  
$` \frac{\partial L_{neg}}{\partial h} = \sum -1 .\left(1-\sigma(-c_{neg}.h)\right).(-1).c_{neg} `$  
$` \frac{\partial L_{neg}}{\partial h} = \sum \left(1-\sigma(-c_{neg}.h)\right).c_{neg} `$  

Where the sum is over **K** negative words.

##### Combined pos and neg words
$` \frac{\partial L}{\partial h} = \left(\sigma(c_{pos}.h) - 1\right).c_{pos} + \sum \left(1-\sigma(-c_{neg}.h)\right).c_{neg} `$  

Where the dimension of the gradient for **h** is [1, embed_dims]. 

In the case of CBOW, `window-1` hidden layers of **h** for `window-1` input context words are averaged to obtain an 
average projection layer for the context words. The gradient for $W_1$ in this case has to be distributed to the `window-1` 
vectors in $W_1$. In this case the gradient should be divided by `window-1`.  
  
E.g. For CBOW:  
$` \frac{\partial L}{\partial h} =\left( \left(\sigma(c_{pos}.h) - 1\right).c_{pos} + \sum \left(1-\sigma(-c_{neg}.h)\right).c_{neg} \right)/(window-1) `$  
<br />
<br />
<br />
#### W.R.T $W_2$, $c_{pos}$ & $c_{neg}$  
$` \frac{\partial L}{\partial c_{pos}} 
= \frac{\partial}{\partial c_{pos}}  \left(-log\left(\sigma\left(c_{pos}.h\right)\right) \right)
= \left(\sigma(c_{pos}.h) - 1\right).h `$  

For `i in [1, K]`:  
$` \frac{\partial L}{\partial c_{neg, i}} 
= \frac{\partial}{\partial c_{neg, i}}  \left(-log\left(\sigma\left(-c_{neg, i}.h\right)\right) \right)
= \left(1 - \sigma(-c_{neg, i}.h)\right).h `$  
  

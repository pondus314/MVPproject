# Zero-Shot Semantic Segmentation with CLIP

Considering the results obtained from the ZS3 model proposed by Bucher, Vu, Cord and Péres and the new capabilities shown by the CLIP model encodings, we aim to evaluate the result of using the CLIP model embeddings in the ZS3 model. The CLIP model embeddings should be more representative than the word2vec embedding used by ZS3 because the CLIP embeddings have been trained in a model that uses both words and images. Therefore, this project aims to improve the model with the use of these features and compare the results.

## Our Demo Colab
[Zero-Shot Semantic Segmentation with CLIP - Colab](https://colab.research.google.com/drive/1e2DN7cOE9gFfwyJvRWlt3nOr-lZQRFRQ)  

## Our trained models
[Trained models](https://drive.google.com/drive/folders/1a8FyhUM6eaNulmyHkn5NjdtzOeI2jNnH?usp=sharing)  

## Original ZS3 Paper
![](./teaser.png)

[Zero-Shot Semantic Segmentation](https://arxiv.org/pdf/1906.00817.pdf)  
 [Maxime Bucher](https://maximebucher.github.io/), [Tuan-Hung Vu](https://tuanhungvu.github.io/) , [Matthieu Cord](http://webia.lip6.fr/~cord/), [Patrick Pérez](https://ptrckprz.github.io/)  
 valeo.ai, France  
 Neural Information Processing Systems (NeurIPS) 2019

[Paper](https://arxiv.org/pdf/1906.00817.pdf):

```
@inproceedings{bucher2019zero,
  title={Zero-Shot Semantic Segmentation},
  author={Bucher, Maxime and Vu, Tuan-Hung and Cord, Mathieu and P{\'e}rez, Patrick},
  booktitle={NeurIPS},
  year={2019}
}
```


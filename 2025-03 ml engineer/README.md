## Install dependencies

I assume you have already installed `python 3.11+` and `pip`.  
In final variant I used Catboost, pandas and scikit-learn metrics so we need those packages to be installed.

```
pip install pandas catboost scikit-learn 
```

---

## Research & Evaluation

I've tested few approaches. 
I've tried BERT, fine-tuned it on our data and got 0.71 f1-score. It's fine, we can work with it.  
But fine-tuning takes a lot of time on free google colab, so I decided to experiment with some classical ML.

<br>

So I thought about Gradient Boosting. And CatBoost is a good choice. It provides text preprocessing. It uses techniques like BoW or BM-25(TF-IDF modification) for text embeddings and then applies gradient boosting algorithm on top of these embeddings.

And it's really fast. ____**We got 0.75 f1-score!**____ 
It's better than BERT. So I decided to go with it.



<br>

You can check jupyter notebook `evaluation.ipynb` for details.  



---

## Result 
I've chosen CatBoost solution as my final approach. 

I think that in general approach with BERT or any other NN should be better after good fine-tuning, especially with more data.

But with no good hardware Gradient Boosting is still a solid choice. 

<br>

What else could be done to improve performance on this task:
* hyperparameter finetuning -- for both BERT and CatBoost
* changing BERT to another model, bigger(better) one
* feature extraction from data (and here we can add also weak labeling techniques)
* combining several models in ensemble with majority voting

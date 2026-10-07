# Results summary

All scores are on the held-out **test set** (6,600 messages). Labels are our weak labels (see `src/labeling.py`), so scores measure agreement with those rules, not with human judgement. Binary models predict neutral (0) vs toxic (1+2), so their class-1 F1 column is the *toxic* class and class 2 is blank.

## Main results (deduplicated data) vs the paper

| Model | Task | Features | Accuracy | Macro F1 | F1 c0 | F1 c1 | F1 c2 | Paper acc. | Paper F1 c0/c1/c2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| SVM linear kernel | binary | word | 0.970 | 0.970 | 0.967 | 0.972 |  | 0.9565 |  |
| SVM linear kernel (char n-grams) | binary | char | 0.987 | 0.987 | 0.985 | 0.988 |  |  |  |
| SVM polynomial kernel (deg 3) | binary | word | 0.875 | 0.875 | 0.868 | 0.882 |  | 0.9435 |  |
| SVM RBF kernel | binary | word | 0.961 | 0.961 | 0.958 | 0.964 |  | 0.9635 |  |
| Naive Bayes (lexicon counts) | multiclass | lexicon | 0.927 | 0.927 | 0.925 | 0.936 | 0.919 | 0.8500 | 0.90/0.77/0.61 |
| Naive Bayes (TF-IDF) | multiclass | word | 0.877 | 0.872 | 0.896 | 0.863 | 0.857 | 0.9506 | 0.97/0.94/0.87 |
| Naive Bayes (char TF-IDF) | multiclass | char | 0.906 | 0.905 | 0.917 | 0.885 | 0.912 |  |  |
| KNN (K=1) | multiclass | word | 0.645 | 0.619 | 0.708 | 0.579 | 0.570 | 0.9853 | 0.99/0.98/0.95 |
| KNN (K=2) | multiclass | word | 0.632 | 0.577 | 0.714 | 0.521 | 0.496 | 0.9711 | 0.98/0.96/0.91 |
| KNN (K=3) | multiclass | word | 0.642 | 0.597 | 0.714 | 0.539 | 0.539 | 0.9683 | 0.98/0.96/0.91 |
| KNN (K=6) | multiclass | word | 0.595 | 0.516 | 0.693 | 0.410 | 0.446 | 0.8424 | 0.89/0.77/0.67 |
| Random Forest (entropy) | multiclass | word | 0.949 | 0.946 | 0.958 | 0.947 | 0.933 | 0.9910 | 0.99/0.99/0.98 |
| Random Forest (entropy, max_depth=32) | multiclass | word | 0.779 | 0.764 | 0.811 | 0.736 | 0.745 | 0.8775 | 0.91/0.86/0.53 |
| Random Forest (Gini) | multiclass | word | 0.952 | 0.949 | 0.961 | 0.950 | 0.935 | 0.9912 | 0.99/0.99/0.98 |
| Logistic Regression (word+char) | multiclass | word+char | 0.985 | 0.984 | 0.988 | 0.982 | 0.982 |  |  |
| Feedforward NN (game-chat word2vec 100d) | multiclass | gamew2v100 | 0.880 | 0.870 | 0.914 | 0.864 | 0.833 |  |  |
| Feedforward NN (game-chat word2vec 50d) | multiclass | gamew2v50 | 0.843 | 0.829 | 0.894 | 0.804 | 0.789 |  |  |
| Feedforward NN (GloVe 100d) | multiclass | glove100 | 0.857 | 0.852 | 0.875 | 0.839 | 0.843 |  |  |
| Feedforward NN (GloVe 50d) | multiclass | glove50 | 0.820 | 0.812 | 0.848 | 0.788 | 0.800 | 0.9486 |  |
| LSTM (game-chat word2vec 100d) | multiclass | gamew2v100 | 0.939 | 0.936 | 0.947 | 0.944 | 0.917 |  |  |
| LSTM (game-chat word2vec 50d) | multiclass | gamew2v50 | 0.933 | 0.930 | 0.942 | 0.937 | 0.911 |  |  |
| LSTM (GloVe 100d) | multiclass | glove100 | 0.921 | 0.918 | 0.929 | 0.917 | 0.909 | 0.9175 |  |
| LSTM (GloVe 50d) | multiclass | glove50 | 0.918 | 0.917 | 0.923 | 0.920 | 0.907 | 0.9586 |  |

## Side experiment: what duplicates do to the scores

Same models and the same 33k size/balance, but sampled without deduplication, so about 30% of test messages also appear word-for-word in train. The last two columns split the with-dups test set into those *seen* messages and the *unseen* rest: near-perfect on seen messages is memorisation, not skill.

| Model | Acc. dedup | Acc. with dups | Δ acc. | Macro F1 dedup | Macro F1 with dups | Δ macro F1 | With dups: acc. seen | With dups: acc. unseen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SVM linear kernel | 0.970 | 0.976 | +0.006 | 0.970 | 0.976 | +0.006 | 0.996 | 0.967 |
| SVM linear kernel (char n-grams) | 0.987 | 0.986 | -0.001 | 0.987 | 0.986 | -0.001 | 0.997 | 0.981 |
| SVM polynomial kernel (deg 3) | 0.875 | 0.872 | -0.003 | 0.875 | 0.870 | -0.005 | 0.993 | 0.816 |
| SVM RBF kernel | 0.961 | 0.971 | +0.010 | 0.961 | 0.971 | +0.010 | 0.998 | 0.958 |
| Naive Bayes (lexicon counts) | 0.927 | 0.933 | +0.006 | 0.927 | 0.934 | +0.007 | 0.965 | 0.919 |
| Naive Bayes (TF-IDF) | 0.877 | 0.899 | +0.022 | 0.872 | 0.894 | +0.022 | 0.988 | 0.858 |
| Naive Bayes (char TF-IDF) | 0.906 | 0.920 | +0.014 | 0.905 | 0.920 | +0.015 | 0.978 | 0.894 |
| KNN (K=1) | 0.645 | 0.775 | +0.131 | 0.619 | 0.763 | +0.144 | 0.998 | 0.673 |
| KNN (K=2) | 0.632 | 0.753 | +0.121 | 0.577 | 0.731 | +0.154 | 0.981 | 0.648 |
| KNN (K=3) | 0.642 | 0.765 | +0.124 | 0.597 | 0.749 | +0.152 | 0.984 | 0.664 |
| KNN (K=6) | 0.595 | 0.728 | +0.133 | 0.516 | 0.702 | +0.186 | 0.959 | 0.622 |
| Random Forest (entropy) | 0.949 | 0.963 | +0.014 | 0.946 | 0.961 | +0.015 | 0.998 | 0.946 |
| Random Forest (entropy, max_depth=32) | 0.779 | 0.863 | +0.083 | 0.764 | 0.858 | +0.094 | 0.916 | 0.838 |
| Random Forest (Gini) | 0.952 | 0.963 | +0.011 | 0.949 | 0.961 | +0.012 | 0.998 | 0.947 |
| Logistic Regression (word+char) | 0.985 | 0.987 | +0.003 | 0.984 | 0.987 | +0.003 | 0.999 | 0.982 |
| Feedforward NN (game-chat word2vec 100d) | 0.880 | 0.924 | +0.045 | 0.870 | 0.919 | +0.048 | 0.994 | 0.892 |
| Feedforward NN (game-chat word2vec 50d) | 0.843 | 0.899 | +0.056 | 0.829 | 0.889 | +0.060 | 0.989 | 0.857 |
| Feedforward NN (GloVe 100d) | 0.857 | 0.897 | +0.039 | 0.852 | 0.892 | +0.040 | 0.970 | 0.862 |
| Feedforward NN (GloVe 50d) | 0.820 | 0.873 | +0.053 | 0.812 | 0.868 | +0.056 | 0.969 | 0.829 |
| LSTM (game-chat word2vec 100d) | 0.939 | 0.958 | +0.019 | 0.936 | 0.956 | +0.020 | 0.993 | 0.941 |
| LSTM (game-chat word2vec 50d) | 0.933 | 0.950 | +0.016 | 0.930 | 0.947 | +0.017 | 0.989 | 0.932 |
| LSTM (GloVe 100d) | 0.921 | 0.933 | +0.012 | 0.918 | 0.931 | +0.013 | 0.970 | 0.915 |
| LSTM (GloVe 50d) | 0.918 | 0.923 | +0.004 | 0.917 | 0.921 | +0.004 | 0.969 | 0.901 |

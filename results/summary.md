# Results summary

All scores are on the held-out **test set** (6,600 messages). Labels are our weak labels (see `src/labeling.py`), so scores measure agreement with those rules, not with human judgement. Binary models predict neutral (0) vs toxic (1+2), so their class-1 F1 column is the *toxic* class and class 2 is blank.

## Main results (deduplicated data) vs the paper

| Model | Task | Features | Accuracy | Macro F1 | F1 c0 | F1 c1 | F1 c2 | Paper acc. | Paper F1 c0/c1/c2 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| SVM linear kernel | binary | word | 0.838 | 0.836 | 0.821 | 0.851 |  | 0.9565 |  |
| SVM linear kernel (char n-grams) | binary | char | 0.845 | 0.843 | 0.826 | 0.861 |  |  |  |
| SVM polynomial kernel (deg 3) | binary | word | 0.770 | 0.763 | 0.724 | 0.802 |  | 0.9435 |  |
| SVM RBF kernel | binary | word | 0.832 | 0.830 | 0.813 | 0.846 |  | 0.9635 |  |
| Naive Bayes (lexicon counts) | multiclass | lexicon | 0.761 | 0.758 | 0.787 | 0.571 | 0.917 | 0.8500 | 0.90/0.77/0.61 |
| Naive Bayes (TF-IDF) | multiclass | word | 0.757 | 0.762 | 0.776 | 0.681 | 0.827 | 0.9506 | 0.97/0.94/0.87 |
| Naive Bayes (char TF-IDF) | multiclass | char | 0.786 | 0.797 | 0.788 | 0.704 | 0.899 |  |  |
| KNN (K=1) | multiclass | word | 0.589 | 0.567 | 0.662 | 0.494 | 0.546 | 0.9853 | 0.99/0.98/0.95 |
| KNN (K=2) | multiclass | word | 0.582 | 0.530 | 0.674 | 0.438 | 0.476 | 0.9711 | 0.98/0.96/0.91 |
| KNN (K=3) | multiclass | word | 0.584 | 0.542 | 0.669 | 0.434 | 0.522 | 0.9683 | 0.98/0.96/0.91 |
| KNN (K=6) | multiclass | word | 0.555 | 0.486 | 0.657 | 0.357 | 0.445 | 0.8424 | 0.89/0.77/0.67 |
| Random Forest (entropy) | multiclass | word | 0.799 | 0.809 | 0.806 | 0.724 | 0.899 | 0.9910 | 0.99/0.99/0.98 |
| Random Forest (entropy, max_depth=32) | multiclass | word | 0.708 | 0.696 | 0.755 | 0.543 | 0.788 | 0.8775 | 0.91/0.86/0.53 |
| Random Forest (Gini) | multiclass | word | 0.800 | 0.810 | 0.805 | 0.728 | 0.898 | 0.9912 | 0.99/0.99/0.98 |
| Logistic Regression (word+char) | multiclass | word+char | 0.833 | 0.845 | 0.832 | 0.759 | 0.942 |  |  |
| Feedforward NN (game-chat fastText 100d) | multiclass | gameft100 | 0.779 | 0.780 | 0.810 | 0.681 | 0.849 |  |  |
| Feedforward NN (game-chat fastText 50d) | multiclass | gameft50 | 0.747 | 0.739 | 0.803 | 0.627 | 0.787 |  |  |
| Feedforward NN (GloVe 100d) | multiclass | glove100 | 0.732 | 0.734 | 0.761 | 0.622 | 0.820 |  |  |
| Feedforward NN (GloVe 50d) | multiclass | glove50 | 0.702 | 0.699 | 0.740 | 0.570 | 0.787 | 0.9486 |  |
| LSTM (game-chat fastText 100d) | multiclass | gameft100 | 0.824 | 0.833 | 0.832 | 0.746 | 0.921 |  |  |
| LSTM (game-chat fastText 50d) | multiclass | gameft50 | 0.817 | 0.823 | 0.830 | 0.731 | 0.906 |  |  |
| LSTM (GloVe 100d) | multiclass | glove100 | 0.777 | 0.785 | 0.791 | 0.687 | 0.878 | 0.9175 |  |
| LSTM (GloVe 50d) | multiclass | glove50 | 0.769 | 0.774 | 0.787 | 0.666 | 0.870 | 0.9586 |  |

## Side experiment: what duplicates do to the scores

Same models and the same 33k size/balance, but sampled without deduplication, so about 30% of test messages also appear word-for-word in train. The last two columns split the with-dups test set into those *seen* messages and the *unseen* rest: near-perfect on seen messages is memorisation, not skill.

| Model | Acc. dedup | Acc. with dups | Δ acc. | Macro F1 dedup | Macro F1 with dups | Δ macro F1 | With dups: acc. seen | With dups: acc. unseen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SVM linear kernel | 0.838 | 0.903 | +0.065 | 0.836 | 0.903 | +0.067 | 0.988 | 0.864 |
| SVM linear kernel (char n-grams) | 0.845 | 0.916 | +0.070 | 0.843 | 0.915 | +0.072 | 0.989 | 0.882 |
| SVM polynomial kernel (deg 3) | 0.770 | 0.836 | +0.067 | 0.763 | 0.835 | +0.072 | 0.991 | 0.765 |
| SVM RBF kernel | 0.832 | 0.905 | +0.073 | 0.830 | 0.905 | +0.075 | 0.994 | 0.864 |
| Naive Bayes (lexicon counts) | 0.761 | 0.830 | +0.069 | 0.758 | 0.826 | +0.068 | 0.947 | 0.775 |
| Naive Bayes (TF-IDF) | 0.757 | 0.840 | +0.083 | 0.762 | 0.837 | +0.075 | 0.974 | 0.778 |
| Naive Bayes (char TF-IDF) | 0.786 | 0.854 | +0.068 | 0.797 | 0.855 | +0.059 | 0.951 | 0.809 |
| KNN (K=1) | 0.589 | 0.745 | +0.156 | 0.567 | 0.729 | +0.161 | 0.983 | 0.635 |
| KNN (K=2) | 0.582 | 0.725 | +0.143 | 0.530 | 0.697 | +0.168 | 0.971 | 0.612 |
| KNN (K=3) | 0.584 | 0.740 | +0.155 | 0.542 | 0.720 | +0.179 | 0.979 | 0.629 |
| KNN (K=6) | 0.555 | 0.710 | +0.155 | 0.486 | 0.678 | +0.192 | 0.959 | 0.595 |
| Random Forest (entropy) | 0.799 | 0.882 | +0.082 | 0.809 | 0.883 | +0.073 | 0.992 | 0.831 |
| Random Forest (entropy, max_depth=32) | 0.708 | 0.788 | +0.080 | 0.696 | 0.779 | +0.084 | 0.922 | 0.727 |
| Random Forest (Gini) | 0.800 | 0.882 | +0.082 | 0.810 | 0.882 | +0.072 | 0.992 | 0.830 |
| Logistic Regression (word+char) | 0.833 | 0.909 | +0.076 | 0.845 | 0.911 | +0.067 | 0.996 | 0.868 |
| Feedforward NN (game-chat fastText 100d) | 0.779 | 0.868 | +0.089 | 0.780 | 0.863 | +0.084 | 0.986 | 0.814 |
| Feedforward NN (game-chat fastText 50d) | 0.747 | 0.849 | +0.102 | 0.739 | 0.839 | +0.100 | 0.980 | 0.788 |
| Feedforward NN (GloVe 100d) | 0.732 | 0.828 | +0.096 | 0.734 | 0.824 | +0.089 | 0.964 | 0.765 |
| Feedforward NN (GloVe 50d) | 0.702 | 0.811 | +0.109 | 0.699 | 0.804 | +0.105 | 0.960 | 0.742 |
| LSTM (game-chat fastText 100d) | 0.824 | 0.898 | +0.073 | 0.833 | 0.899 | +0.066 | 0.987 | 0.857 |
| LSTM (game-chat fastText 50d) | 0.817 | 0.891 | +0.074 | 0.823 | 0.890 | +0.067 | 0.981 | 0.849 |
| LSTM (GloVe 100d) | 0.777 | 0.864 | +0.086 | 0.785 | 0.863 | +0.078 | 0.966 | 0.816 |
| LSTM (GloVe 50d) | 0.769 | 0.858 | +0.090 | 0.774 | 0.858 | +0.084 | 0.963 | 0.810 |

# Learning-NLP-with-disaster-tweets
Introduction to Sentiment Analysis with Disaster Tweets. 
This project was created for learning purposes. The objective was to:
- Explore Natural Language Processing (NLP) and text-based deep learning
- Build a model that classifies tweets as disaster-related or not
- Understand how multiple text features (tweet text, keyword, and location) can be combined to improve performance
- Gain practical experience with TensorFlow Hub embeddings and multi-input neural networks

For more in depth explanation about the steps taken to achieve this project, please read the following article https://medium.com/analytics-vidhya/introduction-to-nlp-with-disaster-tweets-3b672a75748c (*note the code has been enhanced since the article*)

## About the Data
Data used for this project can be found here: https://www.kaggle.com/c/nlp-getting-started/data
The dataset includes:
- text → the tweet content
- keyword → keyword describing the tweet (may be missing)
- location → user’s location (may be missing)
- target → 1 if the tweet refers to a real disaster, 0 otherwise

## Requirements
The following libraries are required to run the notebook:
```
  Python == 3.12.12 
  TensorFlow == 2.19 
  TensorFlow Hub == 0.16.1 
  spaCy == 3.8.7 
  SymSpellPy == 6.9.0 
  Sentence-Transformers == 5.1.1 
  Pandas == 2.2.2 
  NumPy == 2.0.2 
  scikit-learn == 1.6.1 
  Matplotlib == 3.10 
  ```
## Evaluation and Metrics
This is a binary classification problem, where the task is to predict whether a tweet refers to a real disaster (1) or not (0).
Since the dataset is fairly balanced, accuracy was chosen as the primary evaluation metric.

The model achieved an accuracy of ~80.3% on the test set, indicating strong generalization performance on unseen data.

During evaluation, the following metrics were used:
- Accuracy — the proportion of correctly classified samples.
- Precision — measures how many of the tweets predicted as disasters were actually disasters.
- Recall — measures how many of the real disaster tweets were correctly identified by the model.
- F1-Score — the harmonic mean of precision and recall, providing a balance between the two.

The model achieved the following overall results on the test set:

|        Metric       |  Score |
| :-----------------: | :----: |
|     **Accuracy**    | 0.8030 |
| **Precision (avg)** | 0.8001 |
|   **Recall (avg)**  | 0.7962 |
|  **F1-Score (avg)** | 0.7978 |

Both training and validation accuracy/loss were plotted to monitor learning progress.
Early stopping was applied on the validation loss to prevent overfitting and ensure optimal performance.

The Detailed Classification Report was as follows:

|   Metric  | Class 0 (Not Disaster) | Class 1 (Disaster) |
| :-------: | :--------------------: | :----------------: |
| Precision |         0.8165         |       0.7837       |
|   Recall  |         0.8446         |       0.7477       |
|  F1-score |         0.8303         |       0.7653       |

The above confusion matrix indicates that the model correctly identifies most disaster-related and non-disaster tweets.
It performs slightly better on non-disaster tweets (Class 0), as shown by the higher recall (0.84).
However, some disaster tweets (Class 1) are still misclassified as non-disasters, suggesting potential improvements could come from better handling of ambiguous or context-dependent tweets.



# Phishing URL Detection Using Artificial Intelligence

With the increasing use of the Internet, Phishing is a prevalent problem in our world that can cause irreparable harm. There is a great need for tools that can detect malicious URLs with legitimate URLs. This paper focuses on implementing Machine Learning Algorithms (Decision Tree, Logistic Regression, Random Forest) and Large Language Models (BERT, Distil BERT, TinyBERT) to find the best model that works on detecting phishing URLs. SMOTE and TfiDF vectorizer were applied to the Machine Learning models, while BERT Tokenizer was applied to the Large Language Models. The PhiUSIIL dataset was used for the paper where all the models achieved Precision, Recall, and F1 score above 99%. Out of the models, BERT and Random Forest had the best performance. The Precision, Recall, and F1-score for BERT were 99.71%, 99.97%, and 99.84% respectively, and for Random Forest were 99.69%, 99.41%, and 99.97% respectively. A user interface was implemented using Flask for the BERT and Random Forest models.

## Acknowledgements

This is a project for the course "COMP8700: Introduction to AI" at the University of Windsor.



## Models Used

#### Machine Learning Models

1. Decision Tree
2. Random Forest
3. Logistic Regression

#### Transformer Models

1. BERT
2. DistilBERT
3. TinyBERT

## Platforms Used

1. Pandas
2. Sklearn
3. Transformer
4. Torch


## Demo

[![Demo](https://img.youtube.com/vi/<VIDEO_ID>/0.jpg)](https://www.youtube.com/watch?v=dCPn07T8xRA&ab_channel=JoshuaPicchioni)

## Deployment

- create a venv in python
- install all the required libraries using "pip install"
- run flask - "flask run"


# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Tobias Sunderdiek created the model. It is LogisticRegression from scikit-learn 1.0.2 with parameters `max_iter=1000` and `random_state=42`. 

## Intended Use

This model should be used to predict if the salary of a person is above or below 50K. It is trained on census bureau data, and input is based on these features. The users are insurances.

## Training Data

The data is based on the census dataset from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income).

Column names where stripped to remove whitespaces, in total there are 32561 rows with data and one header row.

The data set is split by a 80-20 split to break it into a train and test set. Stratification was done one the target feature `salary`. To use the data during trainig and testing, categorical features where encoded with a OneHotEncoder with parameters `parse=False` and `handle_unknown="ignore"`. The label is encoded with a LabelBinarizer with no parameters.
## Evaluation Data

For evaluation, the same base dataset and the same data pre-processing steps where used as in `Training Data`. The only difference is the size of the resulting dataset, which is 20% of the whole dataset in case of evaluation.

## Metrics
_Please include the metrics used and your model's performance on those metrics._

## Ethical Considerations

## Caveats and Recommendations
very small dataset, few samples

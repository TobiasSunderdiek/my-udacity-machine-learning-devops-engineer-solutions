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

The overall metrics of the model are Precision with 0.7297, Recall with 0.2755 and Fbeta with 0.3999.

## Ethical Considerations

The dataset includes, and the model is trained on features like e.g. `sex`, `native-country`, `race`. These features should not be included in a decision in which equal opportunity is desired because they may bias the decision maker and may lead to unfair decisions. Therefore, these features should also be excluded from the model, to try to avoid bias.

## Caveats and Recommendations
The values of the metrics for some of the slices range from 0.0 (e.g. `education: 1st-4th`) to 1.0 (e.g. `native-country: Hungary`) or nearly in-between (e.g. `education: 12th`) and differ a lot from the overall metrics. The dataset is unbalanced for different features. E.g., only 18 datapoints with `native-country` Laos exists, from which only 2 have a `salary` >50K. 

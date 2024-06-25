# Model Card

## Model Details
- **Model Type**: Logistic Regression
- **Solver**: `lbfgs`
- **Max Iterations**: 2000
- **Scaling**: StandardScaler

- A scaler was implemented due to warning messages on convergence and max iterations 

## Intended Use
- This model is intended to predict if an individual's annual income exceeds $50K based on census data.

## Training Data
- The training data consists of a subset of the census income dataset, with categorical features encoded and continuous features scaled.-
- **Features**: `age`, `workclass`, `fnlgt`, `education`, `education-num`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`
- **Label**: `salary`

## Evaluation Data
- The evaluation data is a separate subset of the census income dataset used to test the model's performance.

## Metrics
- **Precision**: 0.7262
- **Recall**: 0.6038
- **F1 Score**: 0.6594

## Ethical Considerations
- Ensure the model does not reinforce any biases present in the training data.
- Evaluate the model frequently for fairness.

## Caveats and Recommendations
- The model may not generalize well to populations outside the scope of the training data.
- Performance might vary with different distributions or feature sets.
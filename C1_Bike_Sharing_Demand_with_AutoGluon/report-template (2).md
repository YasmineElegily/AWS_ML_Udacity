# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Yasmine Mohamed Elsaied Elegily

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
I realized I had no negative values so I had no problem submitting my results.

### What was the top ranked model that performed?
WeightedEnsemble_L3

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
The histogram of the 'casual' feature is normally distributed
By parsing the 'datetime column' to year, month and day columns

### How much better did your model preform after adding additional features and why do you think that is?
It didn't change at all. I think that is because the year, month and day are strongly correlated to the datetime feature which does not add to the model any new or helpful information for prediction.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
The score of the model got so much higher, but I am expecting the model is overfitted since I tripled the time limit as the kaggle score got much worse.

### If you were given more time with this dataset, where do you think you would spend more time?
By applying more EDA, and see if I could still do more feature engineering and tune each model's hyperparameters
### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.

model	hpo1	hpo2	hpo3	score

0	initial	time_limit = 600	time_limit = 600	time_limit = 1800	1.80982

1	add_features	presets=best_quality	presets=best_quality	presets=best_quality	1.80696

2	hpo	hyperparameters = default	hyperparameters = default	hyperparameters = multimodal	1.32298

### Create a line plot showing the top model score for the three (or more) training runs during the project.

![model_train_score.png](img/model_train_score.png)


### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.


![model_test_score.png](img/model_test_score.png)

## Summary
The new data that the model is tested upon on kaggle is different from the training data so if the model overfits and just memorized the training data so well. When it is tested with new unseen data it stuggled to make predictions and it's accuracy went down.
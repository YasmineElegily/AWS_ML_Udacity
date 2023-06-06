**NOTE:** This file is a template that you can use to create the README for your project. The **TODO** comments below will highlight the information you should be sure to include.

# Dog Breed Image Classification using AWS SageMaker

This project utilizes AWS and SageMaker services to do an end-to-end Machine Learning project. From data Loading and preprocessing to training, hyperparameter tuning jobs and until the deployment of the model. It also utilizes AWS monitoring services for debugging and reporting by using model profiling and debugging.

## Project Set Up and Installation
**OPTIONAL:** If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to make your `README` detailed and self-explanatory. For instance, here you could explain how to set up your project in AWS and provide helpful screenshots of the process.

## Dataset

### Overview
In this project we will be using the dogImages dataset from Amazon. It's a dog breed classification dataset with 133 classes of different dog breeds.
    
### Access
I accessed the data by running this piece of code

    # Command to download and unzip data
    !wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
    !unzip dogImages.zip
    
Then I uploaded it to S3 by running

    input_data = sagemaker_session.upload_data(path="dogImages", bucket=bucket, key_prefix=prefix)
    

## Hyperparameter Tuning
I chose a pre-trained Resnet CNN and used an Adam optimizer for the training job.

Remember that your README should:
- Include a screenshot of completed training jobs
![successful_training_job.png]

- Logs metrics during the training process
![Log_metrics.png]

- Tune at least two hyperparameters
![Hyperparameter_Tuning_job.png]

- Retrieve the best hyperparameters from all your training jobs

    hyperparameters = {"epochs": 5, "batch_size": 32} # best model parameters
    

## Debugging and Profiling

Set up debugging and profiling rules and hooks and configurations then fitted an estimator with the profiler and debugger configuration parameters with train_model.py as an entry point with debugger and profiler configurations. Then plotted and printed the profiler and debugging outputs

### Results

I found this in these useful insights in the profiler output. 

"Ideally, most of the training time should be spent on the TRAIN and EVAL phases. If TRAIN/EVAL were not specified in the training script, steps will be recorded as GLOBAL. Your training job spent quite a significant amount of time (73.9%) in phase "others". You should check what is happening in between the steps."

"During your training job, the StepOutlier rule was the most frequently triggered. It processed 958 datapoints and was triggered 4 times. Check if there are any bottlenecks (CPU, I/O) correlated to the step outliers."

I don't know yet how to check in between training phases. I also checked for bottlenecks but didn't find any.

## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input
Defined an image predictor class and deployed using inference2.py as an entry point with the best parameters model data.
![Endpoint_InService.png] 
And this is how to query an endpoint with a sample input

    from PIL import Image
    import io

    with open("./dogImages/test/127.Silky_terrier/Silky_terrier_08040.jpg", "rb") as f: 
        image = f.read()
        
    response = predictor.predict(image)

    np.argmax(response, 1) + 1



## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.

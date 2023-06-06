**NOTE:** This file is a template that you can use to create the README for your project. The **TODO** comments below will highlight the information you should be sure to include.

# Inventory Monitoring at Distribution Centers

Inventory monitoring at distribution centers is currently facing significant challenges due to outdated monitoring systems and lack of real-time data. These issues have led to an average increase of 15% in warehousing costs and a 20% decrease in order fulfillment.
Distribution centers often use robots to move objects as a part of their operations. Objects are carried in bins which can contain multiple objects. In this project, we will have to build a model that can count the number of objects in each bin. A system like this can be used to track inventory and make sure that delivery consignments have the correct number of items.

## Project Set Up and Installation
**OPTIONAL:** If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to make your `README` detailed and self-explanatory. For instance, here you could explain how to set up your project in AWS and provide helpful screenshots of the process.

## Dataset

### Overview
The dataset we will use is the Amazon Bin Image Dataset which contains over 500,000 images and metadata from bins of a pod in an operating Amazon Fulfillment Center. The bin images in this dataset are captured as robot units carrying pods as part of normal Amazon Fulfillment Center operations. 
Since the dataset is large, we were provided a subset of it to use in this project. Our data are 5 folders named 1 to 5, each representing the number of objects in each image in that folder.


### Access
In order to download the provided subset of our dataset we need to first have the file 'file_list.json' in our directory. Then we need to execute this piece of code to download and arrange the data.


    import os
    import json
    import boto3
    from tqdm import tqdm

    def download_and_arrange_data():
        s3_client = boto3.client('s3')

        with open('file_list.json', 'r') as f:
            d=json.load(f)

        for k, v in d.items():
            print(f"Downloading Images with {k} objects")
            directory=os.path.join('train_data', k)
            if not os.path.exists(directory):
                os.makedirs(directory)
            for file_path in tqdm(v):
                file_name=os.path.basename(file_path).split('.')[0]+'.jpg'
                s3_client.download_file('aft-vbi-pds', os.path.join('bin-images', file_name),
                                 os.path.join(directory, file_name))

    download_and_arrange_data()
    
Then we split the data into train, validation and test data files and upload it by running this:

    !aws s3 cp train_data s3://yasproject3/train_data --recursive
    !aws s3 cp valid_data s3://yasproject3/valid_data --recursive
    !aws s3 cp test_data s3://yasproject3/test_data --recursive

## Model Training
**TODO**: What kind of model did you choose for this experiment and why? Give an overview of the types of hyperparameters that you specified and why you chose them. Also remember to evaluate the performance of your model.

I chose to fine-tune a pre-trained resnet model to do transfer learning and not need to train or use much resources. I chose to use Adam optimizer so I only specified the batch size and the number of epochs since the learning rate is automatically optimized.

## Machine Learning Pipeline

i.Data preparation:
Download a subset of the data.
Preprocess the data.
Upload train, test, and validation files to an S3 bucket so SageMaker use them for training.

ii.Write Model Training script:
Read, load, and preprocess our training, testing and validation data.
Choosing the pre-trained model, optimizer and loss function we will be using.

iii.Train our CNN Model with SageMaker:
Define the hyperparameters, instances type and count for training.
Fitting the estimator

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.

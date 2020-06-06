# Predicting-Food-Delivery-Time

The entire world is transforming digitally and our relationship with technology has grown exponentially over the last few years. We have grown closer to technology, and it has made our life a lot easier by saving time and effort. Today everything is accessible with smartphones — from groceries to cooked food and from medicines to doctors. In this project, we have the data that is a by-product as well as a thriving proof of this growing relationship. 

When was the last time you ordered food online? And how long did it take to reach you?

In this project, we have data from thousands of restaurants in India regarding the time they take to deliver food for online order. As data sciennce aspirants, our goal is to predict the online order delivery time based on the given factors.

# About the data:
#### Size of training set: 11,094 records

#### Size of test set: 2,774 records

## Features:

* Restaurant: A unique ID that represents a restaurant.
* Location: The location of the restaurant.
* Cuisines: The cuisines offered by the restaurant.
* Average_Cost: The average cost for one person/order.
* Minimum_Order: The minimum order amount.
* Rating: Customer rating for the restaurant.
* Votes: The total number of customer votes for the restaurant.
* Reviews: The number of customer reviews for the restaurant.
* Delivery_Time: The order delivery time of the restaurant. (Target Classes) 

# Files in the project: 
* Data_Train.xlsx - training data file
* Data_Test.xlsx - testing data file
* sample_submission.xlsx - format of storing the output variable
* machineHack.ipynb - ipynb file that cleans the data, applies ML algorithm to find the delivery time
* output.xlsx - results on the Data_Test.xlsx stored as per the sample file
* train.py - python file for cleaning the data

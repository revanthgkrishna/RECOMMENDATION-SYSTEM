import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

# Load dataset (Ensure dataset has 'userId', 'movieId', 'rating' columns)
data = pd.read_csv("ratings.csv")

# Define Reader format
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)

# Split into train and test set
trainset, testset = train_test_split(dataset, test_size=0.2)

# Train collaborative filtering model
model = SVD()
model.fit(trainset)

# Make predictions
y_pred = model.test(testset)

# Evaluate model
rmse_score = rmse(y_pred)
print(f"Recommendation System RMSE: {rmse_score:.4f}")
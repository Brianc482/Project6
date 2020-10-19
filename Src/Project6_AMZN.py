import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

dataset = pd.read_csv("https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv")
print("Original dataset:\n",dataset.head())
print("\nShape of data: ",dataset.shape)
print("\nThe keys in our data set:",dataset.keys())


'''Here is the type of data we are dealing with for this project. As we can see there are only two columns, timestamp and value. There are 15831 entried in the dataset that will need to be analyzed'''


time = dataset["timestamp"]
values = dataset["value"]
dataset["timestamp"] = pd.to_datetime(dataset["timestamp"])
print("A description of the data set:")
dataset["timestamp"].describe()

#Here we see a further description of what is contained within our dataset
figure, temp = plt.subplots()
temp.plot(dataset.index, dataset['value'], linestyle="solid", color='cyan')
dates = mdates.YearLocator()
datesFormat = mdates.DateFormatter('%Y')
temp.xaxis.set_major_locator(dates)
temp.xaxis.set_major_formatter(datesFormat)
temp.set(xlabel='Time', ylabel='Volume',title='General overview of the Data')
temp.grid(True)


'''This is a graphical representation generalizing the data within the datasetClearly there are some extreme outliers within the dataset that will be detectedusing anomaly detection methods below'''

tweets = dataset["value"].values
tweetsLen = len(tweets)
tweets = tweets[(tweetsLen//8):tweetsLen-(tweetsLen//8)]
tweetAvg = int(np.floor(tweets.mean()))
print("Average number of tweets in a 30 minute window: ",tweetAvg)
stdDev = tweets.std()
print(f'Standard deviation within the 30 minute window: {stdDev:.2f}')
dataset["scores"] = np.power(((dataset["value"] - tweetAvg)/tweetAvg), 2)
print('\nDescription of the "scores" portion of the data set:')
dataset["scores"].describe()

line1 = 'Number of Tweets'
line2 = 'Average margin'
tempData = dataset.apply(lambda row: np.array(list((row["timestamp"], row["value"], row["scores"]))), axis=1)
tempData = np.array(list(tempData))
plt.figure(figsize=(15, 4))
plt.plot(tempData[:,0], tempData[:,1], color='black', linestyle='solid')
plt.axhspan(tweetAvg - (stdDev / 2), tweetAvg + (stdDev / 2),
             color="yellow", alpha=0.1, zorder=5)
plt.axhspan(tweetAvg - 1, tweetAvg + 1, color="yellow", zorder=6)
plt.title("Tweets about Amazon")
plt.xlabel("Timestamp")
plt.xticks(rotation=30)
plt.ylabel("Number of Tweets")
plt.grid(True)
plt.legend((line1, line2),loc='best')
print("The most activity occurred on:\n")
print(dataset.loc[dataset['value'].idxmax()])

'''Here we can see a much better representation of the amount of tweets and by the date. On March 22th there was a huge spike in the number of Tweets regarding AMZN'''

tempData = tempData[:7 * 48,:]
plt.figure(figsize=(15, 4))
plt.plot(tempData[:,0], tempData[:,1], color='black', linestyle='solid')
plt.axhspan(tweetAvg - (stdDev / 2), tweetAvg + (stdDev / 2), color="yellow", alpha=0.1, zorder=5)
plt.axhspan(tweetAvg - 1, tweetAvg + 1, color="yellow", zorder=6)
plt.title("Tweets about Amazon")
plt.xlabel("Timestamp")
plt.xticks(rotation=30)
plt.ylabel("Number of Tweets")
plt.grid(True)
plt.legend((line1, line2),loc='best')

'''This is a representation of the number of Tweets regarding AMZN over the course of approximately a week. There is a large amount of  fluctuation regarding how many Tweets are posted regarding Amazon'''

figure, temp = plt.subplots()
temp.plot(dataset.index, dataset['value'], color='yellow')
dates = mdates.YearLocator()
datesFormat = mdates.DateFormatter('%Y')
temp.xaxis.set_major_locator(dates)
temp.xaxis.set_major_formatter(datesFormat)
temp.set(xlabel='Time', ylabel='Volume',title='Anomalies Representation')
temp.grid()
for i, row in dataset.iterrows():
    if (row['value'] > 500):
        temp.plot(i, dataset['value'][i], 'D', color='black')
        
plt.grid(True)
plt.show()

'''Here we can see the largest anomalies within the dataset, marked by the black diamonds on the graph.'''

from sklearn.ensemble import IsolationForest
tempData = np.array(dataset)
tempData[:,0] = range(len(dataset))
forest = IsolationForest(contamination=.004)
forestPred = forest.fit_predict(tempData)
dataAnoms = dataset.merge(pd.Series(forestPred, name="anomalies"), left_index=True, right_index=True)
dataAnoms.head()

'''After applying Isolation Forest to the dataset, we can display the features that have been marked as an anomaly. A graphical depiction of  the anomalies over time is shown in the graph below.'''

from sklearn.neighbors import LocalOutlierFactor
anomalies = dataAnoms.loc[dataAnoms["anomalies"] != 1]
anomData = anomalies.apply(lambda row: np.array(list((row["timestamp"], row["value"]))), axis=1)
anomData = np.array(list(anomData))
plt.figure(figsize=(15, 4))
plt.plot(anomData[:,0], anomData[:,1], label="Data")
plt.title("Number of Tweets about Amazon")
plt.xlabel("Dates")
plt.xticks(rotation=30)
plt.ylabel("Number of Tweets")
plt.grid(True)
plt.scatter(anomData[:,0], anomData[:,1], color="red", marker="D", label="anomalies")

tempData = np.array(dataset)
tempData[:,0] = range(len(dataset))
model = LocalOutlierFactor(n_neighbors=50, contamination=.004)
pred = model.fit_predict(tempData)
dataAnom = dataset.merge(pd.Series(pred, name="anomalies"), left_index=True, right_index=True)
anomalies = dataAnom.loc[dataAnom["anomalies"] != 1]
anomData = anomalies.apply(lambda row: np.array(list((row["timestamp"], row["value"]))), axis=1)
anomData = np.array(list(anomData))
plt.figure(figsize=(15, 4))
plt.plot(anomData[:,0], anomData[:,1], label="Data")
plt.title("Number of Tweets about Amazon")
plt.xlabel("Dates")
plt.xticks(rotation=30)
plt.ylabel("Number of Tweets")
plt.scatter(anomData[:,0], anomData[:,1], color="red", marker="D", label="anomalies")

'''Next I applied Local Outlier Factor to the dataset in order to see how one 'Feature Engineering' model might perform in comparison to another.  Here LOF found far fewer anomalies than Isolation Forest, but the majority  of anomalies were found by both models.'''
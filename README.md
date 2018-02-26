# Twitter-RealtimeAnalysis-Pubnub
Sentiment Analysis is an application of opinion mining. It is the process of ‘computationally’ determining whether a piece of writing is positive, negative or neutral to derive the opinion or attitude of a speaker. In the twitter age, where social and commercial interactions have majorly shifter to platforms like twitter, sentiment analysis becomes an inquisitive tool for marketing firms and well for socialogists. Twitter also serves as major platform for any user to express his/her opinions. While, this is healthy for an opionated, politically active generation,a lot of tweets have a hate_speech/racist/offensive element in them.

This project uses PubNub's twitter data stream to analyse user sentiment and the share of hateful,offensive tweets in realtime. For sentiment analysis, Vader Sentiment (Valence Aware Dictionary and sEntiment Reasoner), a simple rule based model is used. 

For topic modelling tweets have been categorized into hateful, offensive or neither classes. Data has been borrowed from https://github.com/t-davidson/hate-speech-and-offensive-language, a repository over around 25000 tweets. Only English tweets are processed for model accuracy and simplicity. Tweet language is detected using relative stop words ratio between 56 nltk corpus languages. A simple Neural Network has been trained using keras with tensorflow backend. Tweets are transformed into Tf-Idf vectors to be used as features in the model. Additional features such as sentiments, URL counts, etc are added, borrowing the results from the paper Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitterhttp. www.aclweb.org/anthology/N16-2013. Details and code for model training can be found in model_train.py. 

Model Architecture:
- Input Layer
- Hidden Layer (ReLU)
- Output Layer (Softmax)

Results are analysed using PubNub's EON framework which helps create amazing live visualizations of realtime data. Pubnub makes life easy with its publisher-subscriber architecture. Results are published over different channels and accessed by the scripts running in Index.html to create spline/bar/donut graphs.

# Running the Code
![alt text](https://github.com/ritiztambi/Twitter-RealtimeAnalysis-Pubnub/blob/master/Training_SS.png)
A SnapShot of model training.

- Run Pubnub_TwitterStream_Analysis.py. Model trains over 10 epochs. Change the location of the data files(highlighted above). 
- Open index.html in a browser.
- Wait for training to finish.
- Push the Start button and Voila!
- Stop the analysis whenever you want. (You may not want to continue your analysis for a long time unless you want smoke coming out of your system. Tensorflow session consumes a lot of memory.

# Result
![alt text](https://github.com/ritiztambi/Twitter-RealtimeAnalysis-Pubnub/blob/master/EON_SS.png)
- The Spline graph on the left plots the ratio of respective sentiments.
- The donut graph depicts the ratio of hate_speech/offensive/neither tweets.
- The bar graphs depict their counts.

# Dependencies
- Python 3
- Keras
- Tensorflow
- Pubnub
- nltk
- vaderSentiment


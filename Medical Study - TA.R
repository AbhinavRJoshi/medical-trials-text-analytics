library(tm)
library(caTools)
library(rpart)
library(rpart.plot)
library(caret)
library(ROCR)
library(randomForest)

trials = read.csv("clinical_trial.csv", stringsAsFactors = FALSE)
trials$trial = as.factor(trials$trial)

corpustitle = Corpus(VectorSource(trials$title))
corpusabstract = Corpus(VectorSource(trials$abstract))
#Creating the corpus from the dataframe

corpustitle = tm_map(corpustitle,tolower)
corpustitle = tm_map(corpustitle, PlainTextDocument)
corpustitle = tm_map(corpustitle, removePunctuation)
corpustitle = tm_map(corpustitle, removeWords, stopwords("english"))
corpustitle = tm_map(corpustitle, stemDocument)


corpusabstract = tm_map(corpusabstract, tolower)
corpusabstract = tm_map(corpusabstract, PlainTextDocument)
corpusabstract = tm_map(corpusabstract, removePunctuation)
corpusabstract = tm_map(corpusabstract, removeWords, stopwords("english"))
corpusabstract = tm_map(corpusabstract, stemDocument)
#Preprocessing the text
#Steps include convering to lowercase, removing punctuations and stop words and stemming words

dtmTitle = DocumentTermMatrix(corpustitle)
dtmTitle = removeSparseTerms(dtmTitle, 0.95)
dtmTitle = as.data.frame(as.matrix(dtmTitle))

dtmAbstract = DocumentTermMatrix(corpusabstract)
dtmAbstract = removeSparseTerms(dtmAbstract, 0.95)
dtmAbstract = as.data.frame(as.matrix(dtmAbstract))
#Creating a Document term matrix, removing words that dont occur in atleast 5% of the observations and
#converting into dataframes

colnames(dtmTitle) = paste0("T", colnames(dtmTitle))
colnames(dtmAbstract) = paste0("A", colnames(dtmAbstract))
#Adding T to words in title and A to words in the abstract to enable differentiation betweent the two
dtm = cbind(dtmTitle,dtmAbstract)
dtm$trial = trials$trial
#converging the two data frames and adding the dependent parameter
set.seed(144)
#To create reproducability of data

spl = sample.split(dtm$trial, SplitRatio = 0.7)
train = subset(dtm, spl == TRUE)
test = subset(dtm,spl == FALSE)
#Creating training and testing sets for our model

table(train$trial)
# Predicting the most frequent outcome which is "Not relevant to trial" (0) gives us a baseline accuracy of 
# 730/1302 = 56.06%

trialglm = glm(trial ~ ., data = train, family = "binomial")
predglm = predict(trialglm, newdata = test, type = "response")
confusionMatrix(predglm > 0.5, test$trial == 1)
#Building a logistic regresssion model gives us an accuracy of 73.3%

trialCart = rpart(trial ~ ., data = train, method = "class")
prp(trialCart)
#Building and plotting a classification tree

predCart = predict(trialCart, newdata = test)[,2]
pred = prediction(predCart,test$trial)
performance(pred,"auc")@y.values
#This model has an AUC value of 0.837
confusionMatrix(predCart >= 0.5, test$trial==1)
# A basic classification tree gives us an accuracy of 75.81%. We will now try to improve this through Cross-Validation of different cp values

control = trainControl(method = "cv",number = 10)
grid = expand.grid(.cp = seq(0,0.2,0.01))
train(trial ~ ., data = train, method = "rpart", trControl= control,tuneGrid = grid)
# From our results we set our cp value as 0.01 and rebuild the model

trialCart2 = rpart(trial ~ ., data = train, method = "class", cp = 0.01)
prp(trialCart2)
predcart2 = predict(trialCart2, newdata = test, type = "class")
confusionMatrix(predcart2, test$trial)
# Since we plotted the same tree with cp = 0.01, our accuracy remains at 75.81%
# We will now try to use Random Forests to improve accuracy further

set.seed(144)
rfTrial = randomForest(trial ~ ., data = train)
predRf = predict(rfTrial, newdata = test, type = "class")
confusionMatrix(predRf, test$trial)
#Using a random forest of 500 trees we improve our prediction accuracy to 83%
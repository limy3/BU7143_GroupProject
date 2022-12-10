#Diabetes Dataset
#Business Problem: efficient allocation of resources
#Statistical Problem: predict potential diabetes individuals
rm(list=ls(all=TRUE))

library(ggplot2)
library(magrittr)
library(stats)
library(caret)
library(gains)
library(rpart)
library(rpart.plot)
library(adabag)

#loading dataset
diabetes<-read.csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

#exploring dataset
View(diabetes)
summary(diabetes)

round(cor(diabetes), 2)
heatmap(cor(diabetes))

##Principal Component Analysis (PCA)
diabetes_eigen<-eigen(cor(diabetes))
diabetes_eigen$values
sum(diabetes_eigen$values)
diabetes_var<-diabetes_eigen$values / sum(diabetes_eigen$values)
diabetes_eigen$vectors

diabetes_pca<-prcomp(diabetes, scale. = TRUE)

diabetes_pca$sdev
diabetes_pca$rotation

#scree plot (1)
qplot(c(1:22), diabetes_var) +
  geom_line() +
  geom_point(size=2)+
  xlab("Principal Component") +
  ylab("Variance Explained") +
  ggtitle("Scree Plot") +
  ylim(0, 1)

#scree plot (2)
screeplot(diabetes_pca, type = "lines")
abline(h = 1.2, lty = 4,lwd = 4, col = "green")

diabetes_pca$x
round(diabetes_pca$rotation, 2)
summary(diabetes_pca)

diabetes_scores<-diabetes_pca$x
plot(diabetes_scores, pch = 19)

#PC1 loadings plot
names.pc1<-ifelse(diabetes_pca$rotation[, 1] > 0, yes = -0.01, no = diabetes_pca$rotation[, 1] - 0.01)
colors.pc1<-ifelse(diabetes_pca$rotation[, 1] > 0, yes = "green2", no = "red2")
par(mar = c(8, 3, 2, 1))
bar.pc1<-barplot(diabetes_pca$rotation[, 1], main = "PC1 Loadings Plot", col = colors.pc1, las = 2, axisnames = FALSE)
abline(h = 0)
text(x = bar.pc1, y = names.pc1, labels = names(diabetes_pca$rotation[, 1]), adj = 1, srt = 90, xpd = TRUE)

#PC2 loadings plot
names.pc2<-ifelse(diabetes_pca$rotation[, 2] > 0, yes = -0.01, no = diabetes_pca$rotation[, 2] - 0.01)
colors.pc2<-ifelse(diabetes_pca$rotation[, 2] > 0, yes = "green2", no = "red2")
par(mar = c(8, 3, 2, 1))
bar.pc2<-barplot(diabetes_pca$rotation[, 2], main = "PC2 Loadings Plot", col = colors.pc2, las = 2, axisnames = FALSE)
abline(h = 0)
text(x = bar.pc2, y = names.pc2, labels = names(diabetes_pca$rotation[, 2]), adj = 1, srt = 90, xpd = TRUE)

#PC3 loadings plot
names.pc3<-ifelse(diabetes_pca$rotation[, 3] > 0, yes = -0.01, no = diabetes_pca$rotation[, 3] - 0.01)
colors.pc3<-ifelse(diabetes_pca$rotation[, 3] > 0, yes = "green2", no = "red2")
par(mar = c(8, 3, 2, 1))
bar.pc3<-barplot(diabetes_pca$rotation[, 3], main = "PC3 Loadings Plot", col = colors.pc3, las = 2, axisnames = FALSE)
abline(h = 0)
text(x = bar.pc3, y = names.pc3, labels = names(diabetes_pca$rotation[, 3]), adj = 1, srt = 90, xpd = TRUE)


#Compare principal components to the original variables using linear regression
fit_1<-lm(Diabetes_binary ~ ., data = diabetes)

components<-cbind(Diabetes_binary = diabetes[, "Diabetes_binary"], diabetes_pca$x[, 1:3]) %>% as.data.frame()
fit_2<-lm(Diabetes_binary ~ ., data = components)

summary(fit_1)$adj.r.squared
summary(fit_2)$adj.r.squared


#partitioning data
set.seed(1)
train.index<-sample(row.names(diabetes), 0.6*dim(diabetes)[1])
valid.index<-setdiff(row.names(diabetes), train.index)
train.df<-diabetes[train.index,]
valid.df<-diabetes[valid.index,]
t(t(names(train.df)))

#new patient information
new_patient<-data.frame(HighBP = 1,
                        HighChol = 1,
                        CholCheck = 1,
                        BMI = 86,
                        Smoker = 0,
                        Stroke = 1,
                        HeartDiseaseorAttack= 1,
                        PhysActivity= 1,
                        Fruits = 0,
                        Veggies = 0,
                        HvyAlcoholConsump= 0,
                        AnyHealthcare = 1,
                        NoDocbcCost = 0,
                        GenHlth = 3.0,
                        MentHlth= 10,
                        PhysHlth = 27.0,
                        DiffWalk = 1,
                        Sex= 0,
                        Age = 12,
                        Education = 7,
                        Income = 8)

train.norm.df<-train.df[, -c(1)]
valid.norm.df<-valid.df[, -c(1)]

#normalising the data
norm.values <- preProcess(train.df[, -c(1)], method=c("center", "scale"))
train.norm.df <- predict(norm.values, train.df[, -c(1)])
valid.norm.df <- predict(norm.values, valid.df[, -c(1)])
new_patient <- predict(norm.values, new_patient)

##kNN model
knn.pred<-class::knn(train = train.norm.df,
                     test = new_patient,
                     cl = train.df$Diabetes_binary, k = 1)

#optimal k
accuracy.df<-data.frame(k = seq(1, 15, 1), overallaccuracy = rep(0, 15))

for(i in 1:15) {
  knn.pred<-class::knn(train = train.norm.df,
                       test = valid.norm.df,
                       cl = train.df$Diabetes_binary, k = i)
  accuracy.df[i, 2]<-confusionMatrix(knn.pred,
                                     as.factor(valid.df$Diabetes_binary))$overall[1]}

accuracy.df

which(accuracy.df[, 2] == max(accuracy.df[, 2]))

knn.pred1<-class::knn(train = train.norm.df,
                      test = valid.norm.df,
                      cl = train.df$Diabetes_binary, k = 15,prob=TRUE)
confusionMatrix(knn.pred1, as.factor(valid.df$Diabetes_binary), positive = "1")

knn.pred2<-class::knn(train = train.norm.df,
                      test = new_patient,
                      cl = train.df$Diabetes_binary, k = 15)

##logistic regression
#use glm() (general linear model) with family = "binomial" to fit a logistic regression.
logit.reg<-glm(Diabetes_binary ~ ., data = train.df, family = "binomial")
summary(logit.reg)

#use predict() with type = "response" to compute predicted probabilities
pred<-predict(logit.reg, valid.df, type = "response")

#first 50 actual and predicted records
round(data.frame(summary(logit.reg)$coefficients, odds = exp(coef(logit.reg))), 5)
round(logit.reg$fitted.values[1:50], 0)


gain<-gains(valid.df$Diabetes_binary, pred, groups = 10)

## lift chart
plot(c(0,gain$cume.pct.of.total*sum(valid.df$Diabetes_binary)) ~
       c(0,gain$cume.obs),
     xlab="#cases", ylab="Cumulative", main="Lift chart (validation dataset", type="l")
lines(c(0,sum(valid.df$Diabetes_binary))~c(0, dim(valid.df)[1]), lty=2)

## decile-wise lift chart
heights<-gain$mean.resp/mean(valid.df$Diabetes_binary)
midpoints<-barplot(heights, names.arg = gain$depth, ylim = c(0, 3),
                   xlab = "Deciles", ylab = "Decile mean/Global mean", main = "Decile-wise lift chart (Validation dataset)")

#add labels to columns
text(midpoints, heights + 0.1, labels=round(heights, 1), cex = 0.8)
valid.df.results = valid.df$Diabetes_binary
confusionMatrix(as.factor(ifelse(pred > 0.5, 1, 0)), as.factor(valid.df.results))
confusionMatrix(as.factor(pred), as.factor(valid.df[,1]))

length((pred))
##kmeans clustering

km<-kmeans(diabetes ,10)

#optimal k for k-means
#Elbow Method for finding the optimal number of clusters
set.seed(123)
# Compute and plot wss for k = 2 to k = 15.
k.max <- 15
data <- dbt.df.norm
wss <- sapply(1:k.max, 
              function(k){kmeans(data, k, nstart=50,iter.max = 15 )$tot.withinss})
wss
plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

#accuracy of k-means clustering
km$betweenss/km$totss
#higher the ratio, the better the performance

## Fitting Naive Bayes Model
# to training dataset
classifier_cl <- naiveBayes(Diabetes_binary ~ ., data = train.df)
classifier_cl

# Predicting on valid data'
y_pred <- predict(classifier_cl, newdata = valid.df)

# Confusion Matrix
cm <- table(valid.df$Diabetes_binary, y_pred)
cm

# Model Evaluation
confusionMatrix(cm)

##random forest
rf<-randomForest(as.factor(Diabetes_binary) ~ ., data = train.df, ntree = 500,
                 mtry = 4, nodesize = 5, importance = TRUE)

#confusion matrix
rf.pred<-predict(rf, valid.df)
confusionMatrix(rf.pred, as.factor(valid.df$Diabetes_binary))
## boosting
train.df$Diabetes_binary<-as.factor(train.df$Diabetes_binary)
str(train.df$Diabetes_binary)
boost<-boosting(Diabetes_binary ~ ., data = train.df, boos = TRUE, mfinal = 100)
pred<-predict(boost, valid.df)
confusionMatrix(as.factor(pred$class), as.factor(valid.df$Diabetes_binary))


##creating a default classification tree
#classification tree - default
diabetes.def<-rpart(Diabetes_binary ~ ., data = train.df, method = "class")
#plot tree
prp(diabetes.def, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10)

#creating a deeper classification tree
#classification tree - deeper
diabetes.deep<-rpart(Diabetes_binary ~ ., data = train.df, method = "class", cp = 0, minsplit = 1)
#count number of leaves
length(diabetes.deep$frame$var[diabetes.deep$frame$var == "<leaf>"])
#plot tree
#prp(diabetes.deep, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10,
#box.col = ifelse(diabetes.deep$frame$var == "<leaf>", 'gray', 'white'))

#confusion matrices and accuracy for the default (small) and deeper (full) classification trees on the training and validation sets
#classifying records in the validation data using both trees

## default (small) tree
#set argument type = "class" in predict() to generate predicted class diabetes status
diabetes.def.point.pred.train<-predict(diabetes.def, train.df, type = "class")
diabetes.def.point.pred.valid<-predict(diabetes.def, valid.df, type = "class")

#generate confusion matrix for training and validation data
confusionMatrix(diabetes.def.point.pred.train, as.factor(train.df$Diabetes_binary))
confusionMatrix(diabetes.def.point.pred.valid, as.factor(valid.df$Diabetes_binary))

## deeper (full) tree
#set argument type = "class" in predict() to generate predicted class diabetes status
diabetes.deep.point.pred.train<-predict(diabetes.deep, train.df, type = "class")
diabetes.deep.point.pred.valid<-predict(diabetes.deep, valid.df, type = "class")

#generate confusion matrix for training and validation data
confusionMatrix(diabetes.deep.point.pred.train, as.factor(train.df$Diabetes_binary))
confusionMatrix(diabetes.deep.point.pred.valid, as.factor(valid.df$Diabetes_binary))

#table of complexity parameter (CP) values and associated tree errors
#tabulating tree error as a function of the complexity parameter (CP)

#argument xval refers to the number of folds to use in rpart's built-in cross-validation procedure
#argument CP sets the smallest value for the complexity parameter

cv.ct<-rpart(Diabetes_binary ~ ., data = train.df, method = "class",
             cp = 0.00001, minsplit = 5, xval = 5)

#use printcp() to print the table
printcp(cv.ct)
table<-as.data.frame(printcp(cv.ct))

#pruning the tree
#prune by lower cp

pruned.classtree<-prune(cv.ct,
                        cp = cv.ct$cptable[which.min(cv.ct$cptable[, "xerror"]), "CP"])
length(pruned.classtree$frame$var[pruned.classtree$frame$var == "<leaf>"])
prp(pruned.classtree, type = 1, extra = 1, split.font = 1, varlen = -10)

#Ensemble
res<- data.frame(ActualClass = valid.df$Diabetes_binary,
                 LRProb = predict(logit.reg, valid.df, type = "response"),
                 LRPred = ifelse(predict(logit.reg, valid.df, type = "response")>0.5, 1, 0),
                 KNNProb = 1-attr(knn.pred1,"prob"),
                 KNNPred = knn.pred1,
                 TRProb = predict(diabetes.def, valid.df)[,1],
                 TRPred = ifelse(predict(diabetes.def, valid.df)[,1]>0.5, 1, 0))

options(digits = 1, scipen = 2)
head(res, 10)

#ensemble model

res$majority <- rowMeans(data.frame(res$LRPred, as.numeric(res$KNNPred),
                res$TRPred))>0.5
res$avg <- rowMeans(data.frame(res$LRProb, res$KNNProb, res$TRPred))
head(res)

options(digits = 7, scipen = 2)
#confusion matrix using majority vote of predicted outcomes
confusionMatrix(factor(res$majority * 1), as.factor(valid.df$Diabetes_binary), positive = "1")
confusionMatrix(factor((res$avg > 0.5)* 1), as.factor(valid.df$Diabetes_binary), positive = "1")


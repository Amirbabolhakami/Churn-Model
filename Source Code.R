rm(list = ls())


install.packages("readr")
install.packages("dplyr")
install.packages("ggplot2") 
install.packages("colorspace")
install.packages("ggcorrplot")
install.packages("factoextra")
install.packages("cluster")
install.packages("fpc")
install.packages("C50")
install.packages('caret')
install.packages("future")
install.packages("e1071")
install.packages("class")
install.packages("RWeka")

#Required Library
library(readr)
library(dplyr)
library(ggplot2)
library(ggcorrplot)
library(factoextra)
library(cluster)
library(fpc)
library(C50)
library(caret)
library(e1071)
library(class)

#Read the Dataset
Dataset <- read_csv("Churn_Modelling.csv")


#statistical information - Numeric Value
for (i in 4:ncol(Dataset)) {
  print(colnames(Dataset[,i]))
  print(summary(Dataset[,i]))
  print("------------------------------------")
}

#statistical information - Nominal Value
table(Dataset$Geography)
table(Dataset$Gender)
table(Dataset$Tenure)
table(Dataset$HasCrCard)
table(Dataset$IsActiveMember)
table(Dataset$Exited)

#Pre-Processing - Null Value
sum(is.na(Dataset))
Dataset1 = na.omit(Dataset)

#Pre-Processing - Select Attribute1           
Dataset1 = Dataset1[,4:ncol(Dataset1)]


#Pre-Processing - Outlier Value
x = boxplot(Dataset1$CreditScore)
max(x$out)
y = boxplot(Dataset1$Age)
min(y$out)
boxplot(Dataset1$Balance)
boxplot(Dataset1$EstimatedSalary)

Dataset1 = Dataset1[Dataset1$Age < min(y$out) | 
                      Dataset1$CreditScore < max(x$out),]



#Data Visualization - Histogram

hist(Dataset1$CreditScore, col = "#00AFBB", xlab = "Credit Score")
hist(Dataset1$Age, col = "#E7B800", xlab = "Age")
hist(Dataset1$Balance, col = "#FC4E07", xlab = "Balance")
hist(Dataset1$EstimatedSalary, col = "blue3", xlab = "Estimated Salary")

#Data Visualization - Barplot

x = as.data.frame(table(Dataset1$Gender))
ggplot(x, aes(x = Var1, y = Freq, color = Var1)) + 
  geom_bar(stat = "identity")

x = as.data.frame(table(Dataset1$Geography))
ggplot(x, aes(x = Var1, y = Freq, color = Var1)) + 
  geom_bar(stat = "identity", color="purple", fill="blue")

x = as.data.frame(table(Dataset1$Tenure))
ggplot(x, aes(x = Var1, y = Freq, color = Var1)) + 
  geom_bar(stat = "identity", color="blue", fill="white")

x = as.data.frame(table(Dataset1$HasCrCard))
ggplot(x, aes(x = Var1, y = Freq, color = Var1)) + 
  geom_bar(stat = "identity", fill="steelblue")

x = as.data.frame(table(Dataset1$IsActiveMember))
ggplot(x, aes(x = Var1, y = Freq, color = Var1)) + 
  geom_bar(stat = "identity", color="blue", fill="brown") +
  geom_text(aes(label= Freq), vjust=-0.3, size=3.5)+
  theme_minimal()

x = as.data.frame(table(Dataset1$Exited))
ggplot(x, aes(x = Var1, y = Freq)) + 
  geom_bar(stat = "identity", color="pink") +
  geom_text(aes(label= Freq), vjust=1.6, size=3.5)+
  theme_minimal()



#correlation coefficient - Numeric Attribute

FS_Pearson_Num_Att =cor(Dataset1[,c(1,4, 6, 7, 10, 11)], method = "pearson")
FS_Pearson_Num_Att = round(FS_Pearson_Num_Att, digits = 3)
ggcorrplot(FS_Pearson_Num_Att)
FS_Pearson_Num_Att

#correlation coefficient - Nominal Attribute
Dataset1$Geography = as.numeric(as.factor(Dataset1$Geography))
Dataset1$Gender = as.numeric(as.factor(Dataset1$Gender))
FS_Pearson_Nom_Att =cor(Dataset1[,c(2, 3, 5, 8, 9, 11)], method = "pearson")
FS_Pearson_Nom_Att = round(FS_Pearson_Nom_Att, digits = 3)
ggcorrplot(FS_Pearson_Nom_Att)
FS_Pearson_Nom_Att

#Pre-Processing - Select Attribute 2
Dataset1 = Dataset1[,-8]

#Clustreing Methods ---- Kmeans
K_Model = kmeans(Dataset1[,-10], 4)
print(K_Model)
print(K_Model$size)
fviz_cluster(K_Model, data = Dataset1[,-10])

#Discretization --- Numeric to Nominal

Dataset1 = as.data.frame(Dataset1)
Dataset1[,2] = as.factor(as.numeric(Dataset1[,2]))
Dataset1[,3] = as.factor(as.numeric(Dataset1[,3]))
Dataset1[,5] = as.factor(Dataset1[,5])
Dataset1[,8] = as.factor(Dataset1[,8])
Dataset1[,10] = as.factor(Dataset1[,10])

#Classification Model - Decision Tree
result_Dataset = as.data.frame(matrix(0, ncol = 7))
colnames(result_Dataset) = c("Fold", "FoldNum", "Acc", "Precision", 
                             "Recall", "sensitivity", "Specificity")

j = 6
k = 1
while (j <= 10) {
  FD = j
  
  j = j + 2
  folds = createFolds(Dataset1$Exited, k = FD, list = TRUE, returnTrain = TRUE)
  for (i in 1:FD) {
    Held_Out_Indices = folds[[i]]
    train_Set = Dataset1[Held_Out_Indices,]
    test_set = Dataset1[-Held_Out_Indices,]
    TrainedClassifier <- C5.0(Exited~., data = train_Set)
    TrainedClassifier
    plot(TrainedClassifier)
    Predictions <- predict(TrainedClassifier, newdata=test_set)
    #performance evaluation - Decision Tree
    cm <- confusionMatrix(test_set$Exited, Predictions)
    result_Dataset[k,1] = FD
    result_Dataset[k, 2] = i
    result_Dataset[k, 3]= round(cm[["overall"]][["Accuracy"]], 2)
    result_Dataset[k, 4]= round(cm[["byClass"]][["Precision"]], 2)
    result_Dataset[k, 5]= round(cm[["byClass"]][["Recall"]], 2)
    result_Dataset[k, 6]= round(cm[["byClass"]][["Sensitivity"]], 2)
    result_Dataset[k, 7]= round(cm[["byClass"]][["Specificity"]], 2)
    k = k + 1
  }
}

Final_Result = as.data.frame(aggregate(result_Dataset[,3:7], 
                                       list(result_Dataset$Fold), mean))
barplot(Final_Result$Acc, Final_Result$Group.1, xlab = "Accuracy")

ggplot(Final_Result, aes(x = Precision, y = Group.1)) + 
  geom_bar(stat = "identity", color="pink") +
  geom_text(aes(label= Group.1), vjust=1.6, size=3.5)+
  theme_minimal()

ggplot(Final_Result, aes(x = Acc, y = Group.1)) + 
  geom_bar(stat = "identity", color="pink") +
  geom_text(aes(label= Group.1), vjust=1.6, size=3.5)+
  theme_minimal()

ggplot(Final_Result, aes(x = sensitivity, y = Group.1)) + 
  geom_bar(stat = "identity", color="pink") +
  geom_text(aes(label= Group.1), vjust=1.6, size=3.5)+
  theme_minimal()

ggplot(Final_Result, aes(x = Specificity, y = Group.1)) + 
  geom_bar(stat = "identity", color="pink") +
  geom_text(aes(label= Group.1), vjust=1.6, size=3.5)+
  theme_minimal()

write.csv(result_Dataset, "Decision Tree Result.csv")
write.csv(Final_Result, "Final Result of Decision tree.csv")


#Classification Model - Knn
result_Dataset = as.data.frame(matrix(0, ncol = 7))
colnames(result_Dataset) = c("Fold", "FoldNum", "Acc", "Precision", 
                             "Recall", "sensitivity", "Specificity")

j = 6
k = 1
while (j <= 10) {
  FD = j
  
  j = j + 2
  folds = createFolds(Dataset1$Exited, k = FD, list = TRUE, returnTrain = TRUE)
  for (i in 1:FD) {
    Held_Out_Indices = folds[[i]]
    train_Set = Dataset1[Held_Out_Indices,]
    test_set = Dataset1[-Held_Out_Indices,]
    TrainedClassifier <- knn(train_Set,test_set,cl= train_Set$Exited, k = 20)
    TrainedClassifier
    #performance evaluation - KNN
    cm <- confusionMatrix(test_set$Exited, TrainedClassifier)
    result_Dataset[k,1] = FD
    result_Dataset[k, 2] = i
    result_Dataset[k, 3]= round(cm[["overall"]][["Accuracy"]], 2)
    result_Dataset[k, 4]= round(cm[["byClass"]][["Precision"]], 2)
    result_Dataset[k, 5]= round(cm[["byClass"]][["Recall"]], 2)
    result_Dataset[k, 6]= round(cm[["byClass"]][["Sensitivity"]], 2)
    result_Dataset[k, 7]= round(cm[["byClass"]][["Specificity"]], 2)
    k = k + 1
  }
}

Final_Result = as.data.frame(aggregate(result_Dataset[,3:7], 
                                       list(result_Dataset$Fold), mean))

ggplot(Final_Result, aes(x = Precision, y = Group.1)) + 
  geom_bar(stat = "identity", color="pink") +
  geom_text(aes(label= Group.1), vjust=1.6, size=3.5)+
  theme_minimal()

ggplot(Final_Result, aes(x = Acc, y = Group.1)) + 
  geom_bar(stat = "identity", color="pink") +
  geom_text(aes(label= Group.1), vjust=1.6, size=3.5)+
  theme_minimal()

ggplot(Final_Result, aes(x = sensitivity, y = Group.1)) + 
  geom_bar(stat = "identity", color="pink") +
  geom_text(aes(label= Group.1), vjust=1.6, size=3.5)+
  theme_minimal()

ggplot(Final_Result, aes(x = Specificity, y = Group.1)) + 
  geom_bar(stat = "identity", color="pink") +
  geom_text(aes(label= Group.1), vjust=1.6, size=3.5)+
  theme_minimal()

write.csv(result_Dataset, "KNN Result.csv")
write.csv(Final_Result, "Final Result of KNN.csv")



#Classification Model - Naive Bayes
result_Dataset = as.data.frame(matrix(0, ncol = 7))
colnames(result_Dataset) = c("Fold", "FoldNum", "Acc", "Precision", 
                             "Recall", "sensitivity", "Specificity")

j = 6
k = 1
while (j <= 10) {
  FD = j
  
  j = j + 2
  folds = createFolds(Dataset1$Exited, k = FD, list = TRUE, returnTrain = TRUE)
  for (i in 1:FD) {
    Held_Out_Indices = folds[[i]]
    train_Set = Dataset1[Held_Out_Indices,]
    test_set = Dataset1[-Held_Out_Indices,]
    TrainedClassifier <- naiveBayes(Exited~., data = train_Set)
    TrainedClassifier
    Predictions <- predict(TrainedClassifier, newdata=test_set)
    #performance evaluation - Naive Bayes
    cm <- confusionMatrix(test_set$Exited, Predictions)
    result_Dataset[k,1] = FD
    result_Dataset[k, 2] = i
    result_Dataset[k, 3]= round(cm[["overall"]][["Accuracy"]], 2)
    result_Dataset[k, 4]= round(cm[["byClass"]][["Precision"]], 2)
    result_Dataset[k, 5]= round(cm[["byClass"]][["Recall"]], 2)
    result_Dataset[k, 6]= round(cm[["byClass"]][["Sensitivity"]], 2)
    result_Dataset[k, 7]= round(cm[["byClass"]][["Specificity"]], 2)
    k = k + 1
  }
}

Final_Result = as.data.frame(aggregate(result_Dataset[,3:7], 
                                       list(result_Dataset$Fold), mean))

ggplot(Final_Result, aes(x = Precision, y = Group.1)) + 
  geom_bar(stat = "identity", color="pink") +
  geom_text(aes(label= Group.1), vjust=1.6, size=3.5)+
  theme_minimal()

ggplot(Final_Result, aes(x = Acc, y = Group.1)) + 
  geom_bar(stat = "identity", color="pink") +
  geom_text(aes(label= Group.1), vjust=1.6, size=3.5)+
  theme_minimal()

ggplot(Final_Result, aes(x = sensitivity, y = Group.1)) + 
  geom_bar(stat = "identity", color="pink") +
  geom_text(aes(label= Group.1), vjust=1.6, size=3.5)+
  theme_minimal()

ggplot(Final_Result, aes(x = Specificity, y = Group.1)) + 
  geom_bar(stat = "identity", color="pink") +
  geom_text(aes(label= Group.1), vjust=1.6, size=3.5)+
  theme_minimal()

write.csv(result_Dataset, "Naive Bayes Result.csv")
write.csv(Final_Result, "Final Result of Naive Bayes.csv")


#Classification Model - Lazy Model
result_Dataset = as.data.frame(matrix(0, ncol = 7))
colnames(result_Dataset) = c("Fold", "FoldNum", "Acc", "Precision", 
                             "Recall", "sensitivity", "Specificity")

j = 6
k = 1
while (j <= 10) {
  FD = j
  
  j = j + 2
  folds = createFolds(Dataset1$Exited, k = FD, list = TRUE, returnTrain = TRUE)
  for (i in 1:FD) {
    Held_Out_Indices = folds[[i]]
    train_Set = Dataset1[Held_Out_Indices,]
    test_set = Dataset1[-Held_Out_Indices,]
    TrainedClassifier <- C5.0(x = train_Set[, -10],
                              y = train_Set$Exited, rules = TRUE)
    TrainedClassifier
    Predictions <- predict(TrainedClassifier, newdata=test_set)
    #performance evaluation - Rule Based
    cm <- confusionMatrix(test_set$Exited, Predictions)
    result_Dataset[k,1] = FD
    result_Dataset[k, 2] = i
    result_Dataset[k, 3]= round(cm[["overall"]][["Accuracy"]], 2)
    result_Dataset[k, 4]= round(cm[["byClass"]][["Precision"]], 2)
    result_Dataset[k, 5]= round(cm[["byClass"]][["Recall"]], 2)
    result_Dataset[k, 6]= round(cm[["byClass"]][["Sensitivity"]], 2)
    result_Dataset[k, 7]= round(cm[["byClass"]][["Specificity"]], 2)
    k = k + 1
  }
}

Final_Result = as.data.frame(aggregate(result_Dataset[,3:7], 
                                       list(result_Dataset$Fold), mean))

ggplot(Final_Result, aes(x = Precision, y = Group.1)) + 
  geom_bar(stat = "identity", color="pink") +
  geom_text(aes(label= Group.1), vjust=1.6, size=3.5)+
  theme_minimal()

ggplot(Final_Result, aes(x = Acc, y = Group.1)) + 
  geom_bar(stat = "identity", color="pink") +
  geom_text(aes(label= Group.1), vjust=1.6, size=3.5)+
  theme_minimal()

ggplot(Final_Result, aes(x = sensitivity, y = Group.1)) + 
  geom_bar(stat = "identity", color="pink") +
  geom_text(aes(label= Group.1), vjust=1.6, size=3.5)+
  theme_minimal()

ggplot(Final_Result, aes(x = Specificity, y = Group.1)) + 
  geom_bar(stat = "identity", color="pink") +
  geom_text(aes(label= Group.1), vjust=1.6, size=3.5)+
  theme_minimal()

write.csv(result_Dataset, "Rule Based Result.csv")
write.csv(Final_Result, "Final Result of Rule Based.csv")

















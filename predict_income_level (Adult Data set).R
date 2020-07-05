
## Student Name: Chanpreet Singh , Roll number : 101128629 , MIT-DIGITAL MEDIA (DATA SCIENCE)
#############Author: Chanpreet Singh, School of IT ##################################################
#############Special Thanks to Prof Dr. Omair Shafiq for helping in almost everypart of the project#####
#PREDICTION OF INCOME LEVEL AMONG INDIVIDUALS USING US CENSUS DATA.
# Install library packages - e1071, ggplot2, reshape2, randomForest, rpart, rattle

library(tictoc)
tic()
library(e1071)  # for implementing Naives Bayes classifier
library(ggplot2) # for different plots
library(reshape2)
library(randomForest)  # for implementin random forest classifier
library(rpart)
library(rattle)
library(corrplot)
library(gridExtra)
library(MASS)
library(gbm)
library(glmnet)
library(nnet)
library(class)
library(xtable)  
library(parallelSVM)
library(FastKNN)
library(caret)



########################### testing and training data ####################################################################



file_name="/Users/apple/Downloads/adult.csv"
adult.data<- read.table(file = file_name, header = TRUE, sep = ",", 
                        strip.white = TRUE, stringsAsFactors = TRUE,
                        col.names=c("age","workclass","fnlwgt","education","educationnum","maritalstatus",
                                    "occupation","relationship","race","sex","capitalgain","capitalloss",
                                    "hoursperweek","nativecountry","income")
)
adult<- adult.data # make a copy of the data 

# Exploratory Data Analysis
## a. Structure 
dim(adult.data)
str(adult.data)

# collapse the factor levels and recode level with no name (coded as ?in original data) to missing
        levels(adult.data$workclass)
levels(adult.data$workclass)<- c("misLeveL","FedGov","LocGov","NeverWorked","Private","SelfEmpInc",
                                 "SelfEmpNotInc","StateGov","NoPay","workclass")
           levels(adult.data$education)
levels(adult.data$education)<- list(presch=c("Preschool"), primary=c("1st-4th","5th-6th"),
                                    upperprim=c("7th-8th"), highsch=c("9th","Assoc-acdm","Assoc-voc","10th"),
                                    secndrysch=c("11th","12th"), graduate=c("Bachelors","Some-college","Prof-school"),
                                    master=c("Masters"), phd=c("Doctorate"))
   levels(adult.data$maritalstatus)<- list(divorce=c("Divorced","Separated"), 
                                        married=c("Married-AF-spouse","Married-civ-spouse","Married-spouse-absent"),
                                        notmarried=c("Never-married"), widowed=c("Widowed"))

       levels(adult.data$occupation) # missing level name coded as `?`
levels(adult.data$occupation)<- list(misLevel=c("?"), clerical=c("Adm-clerical"), 
                                     lowskillabr=c("Craft-repair","Handlers-cleaners","Machine-op-inspct",
                                                   "Other-service","Priv-house-serv","Prof-specialty",
                                                   "Protective-serv"),
                                     highskillabr=c("Sales","Tech-support","Transport-moving","Armed-Forces"),
                                     agricultr=c("Farming-fishing")
)

  levels(adult.data$relationship)<- list(husband=c("Husband"), wife=c("Wife"), outofamily=c("Not-in-family"),
                                       unmarried=c("Unmarried"), relative=c("Other-relative"), 
                                       ownchild=c("Own-child"))

       levels(adult.data$nativecountry)<- list(misLevel=c("?","South"),SEAsia=c("Vietnam","Laos","Cambodia","Thailand"),
                                        Asia=c("China","India","HongKong","Iran","Philippines","Taiwan"),
                                        NorthAmerica=c("Canada","Cuba","Dominican-Republic","Guatemala","Haiti",
                                                       "Honduras","Jamaica","Mexico","Nicaragua","Puerto-Rico",
                                                       "El-Salvador","United-States"),
                                        SouthAmerica=c("Ecuador","Peru","Columbia","Trinadad&Tobago"),
                                        Europe=c("France","Germany","Greece","Holand-Netherlands","Italy",
                                                 "Hungary","Ireland","Poland","Portugal","Scotland","England",
                                                 "Yugoslavia"),
                                        PacificIslands=c("Japan","France"),
                                        Oceania=c("Outlying-US(Guam-USVI-etc)")
)
levels(adult.data$income)

# check for missing values
        colSums(is.na(adult.data)) # missing values in, education(11077) occupation(4066) and native.country(20)
str(adult.data)

## Missing data imputation

            str(adult.data) # generally it is advisable not to impute the categorical missing values, if they are less than they should be removed

#levels(adult.data$income)<- list(less50K=c("<=50K"), gr50K=c(">50K"))
library(VIM)
           aggr_plot <- aggr(adult.data, col=c('orange','purple'), numbers=TRUE, sortVars=TRUE, 
                  labels=names(adult.data), cex.axis=.7, gap=3, 
                  ylab=c("Histogram of missing data","Pattern")
)
library(missForest)
     imputdata<- missForest(adult.data) 
# check imputed values
imputdata$ximp
# assign imputed values to a data frame
adult.cmplt<- imputdata$ximp
df.master<- adult.cmplt # save a copy

set.seed(1234)
ratio = sample(1:nrow(adult.cmplt), size = 0.25*nrow(adult.cmplt))
test = adult.cmplt[ratio,] #Test dataset 25% of total
train = adult.cmplt[-ratio,] #Train dataset 75% of total

dim(train)
dim(test)
str(train)

##################################### CORELATION VALUES #########################################################################

#cor(clean_train[sapply(clean_train, function(x) !is.factor(x))])
X = data.matrix(train)
corr_matrix = cor(X)
corrplot(corr_matrix)
corrplot(corr_matrix,
         method = 'ellipse',
         type = "full")


#######################################   DV ######################################################################################

ggplot(train, aes(x = age, color = income, fill = income)) + 
  geom_density(alpha = 0.8) + scale_fill_manual(values=c("#F8B83F", "#2ECC71")) +
  theme(panel.border = element_rect(fill=NA,color="black", size=0.5, 
                                    linetype="solid")) +
  scale_colour_manual(values=c("purple","purple"))
labs(x = "Age", y = "Density", title = "People who are older earn more",
     subtitle = "Density plot")


ggplot(train, aes(x = workclass, fill = income, color = income)) +
  geom_bar(alpha = 0.8, position = "fill") +
  coord_flip() + scale_fill_manual(values=c("#F8B83F", "#2ECC71")) +
  theme(panel.border = element_rect(fill=NA,color="black", size=0.5, 
                                    linetype="solid")) +
  scale_colour_manual(values=c("purple","purple"))
labs(x = "Working Class", y = "Proportion", title = "Income bias based on working class",
     subtitle = "Stacked bar plot")

ggplot(train, aes(x = education, fill = income, color = income)) +
  geom_bar(alpha = 0.8, position = "fill") +
  coord_flip() + scale_fill_manual(values=c("#F8B83F", "#2ECC71")) +
  theme(panel.border = element_rect(fill=NA,color="black", size=0.5, 
                                    linetype="solid")) +
  scale_colour_manual(values=c("purple","purple"))
labs(x = "Education Level", y = "Proportion", title = "People with more education earn more",
     subtitle = "Stacked bar plot")

ggplot(train, aes(x = maritalstatus, fill = income, color = income)) +
  geom_bar(alpha = 0.8, position = "fill") +
  coord_flip() + scale_fill_manual(values=c("#F8B83F", "#2ECC71")) +
  theme(panel.border = element_rect(fill=NA,color="black", size=0.5, 
                                    linetype="solid")) +
  scale_colour_manual(values=c("purple","purple"))
labs(x = "Marital Status", y = "Proportion", title = "Married people tend to earn more",
     subtitle = "Stacked bar plot")

ggplot(train,aes(x = race, fill = income, color = income)) +
  geom_bar(alpha = 0.8, position = "fill") +
  coord_flip() + scale_fill_manual(values=c("#F8B83F", "#2ECC71")) +
  theme(panel.border = element_rect(fill=NA,color="black", size=0.5, 
                                    linetype="solid")) +
  scale_colour_manual(values=c("purple","purple"))
labs(x = "Race", y = "Proportion", title = "There is a bias in income based on race",
     subtitle = "Stacked bar plot")

ggplot(train,aes(x = occupation, fill = income, color = income)) +
  geom_bar(alpha = 0.8, position = "fill") +
  coord_flip() + scale_fill_manual(values=c("#F8B83F", "#2ECC71")) +
  theme(panel.border = element_rect(fill=NA,color="black", size=0.5, 
                                    linetype="solid")) +
  scale_colour_manual(values=c("purple","purple"))
labs(x = "Occupation", y = "Proportion", title = "There is a bias in the income based on occupation",
     subtitle = "Stacked bar plot")

ggplot(train, aes(x = occupation, fill = income, color = income)) +
  geom_bar(alpha = 0.8, position = "fill") +
  coord_flip() + scale_fill_manual(values=c("#F8B83F", "#2ECC71")) +
  theme(panel.border = element_rect(fill=NA,color="black", size=0.5, 
                                    linetype="solid")) +
  scale_colour_manual(values=c("purple","purple"))
labs(x = "Occupation", y = "Proportion", title = "There is a bias in the income based on occupation",
     subtitle = "Stacked bar plot")


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

qplot(income, data = adult.cmplt, fill = occupation) + facet_grid (. ~ occupation)

qplot(income, data = adult.cmplt, fill = education) + facet_grid (. ~ education)




#######################################  DATA VISUALIZATION ##########################################################################

boxplot (age ~ incomelevel, data = clean_train, 
         main = "Age distribution for different income levels",
  xlab = "Income Levels", ylab = "Age", col = c("green","orange"))

boxplot (educationnum ~ incomelevel, data = clean_train, 
    main = "Highest Level of education distribution for different income levels",
         xlab = "Income Levels", ylab = "EducationNum", col = c("green","orange"))

boxplot (hoursperweek ~ incomelevel, data = clean_train, 
              main = "Hours per week distribution for different income levels",
         xlab = "Income Levels", ylab = "Hours per week", col = c("green","orange"))

boxplot (fnlwgt ~ incomelevel, data = clean_train, 
           main = "Final weight distribution for different income levels",
         xlab = "Income Levels", ylab = "Final weight", col = c("green","orange"))

#Plot density for each attribute using lattice
par(mfrow=c(3,5))
for(i in 1:15) {
  plot(density(X[,i]), main=colnames(X)[i])
}
par(mfrow=c(1,1))

ggplot(clean_train, aes(x = sex,fill=incomelevel))+ geom_bar(stat="count",position = "dodge")+
      ggtitle("Distribution of incomelevels w.r.t sex ") + scale_fill_brewer(palette = 'Set2')

# Skipped bar plots for educationnum, hoursperweek and age. Box plots are already done for these variables
           #ggplot(clean_train, aes(x = educationnum,fill=incomelevel))+ geom_bar(stat="count",position = "dodge")+
# scale_fill_brewer(palette = 'Set1')+ggtitle("Distribution of incomelevel w.r.t educationnum")

ggplot(clean_train, aes(x = workclass,fill=incomelevel))+ geom_bar(stat="count", position = "dodge")+
  scale_fill_brewer(palette = 'Set2')+ggtitle("Distribution of incomelevels w.r.t workclass")

#ggplot(clean_train, aes(x = occupation, fill=incomelevel))+ geom_bar(stat="count",position = "dodge")+
#  scale_fill_brewer(palette = 'Set1')+ggtitle("Distribution of incomelevels w.r.t occupation")

ggplot(clean_train, aes(x = maritalstatus, fill=incomelevel))+ geom_bar(stat="count",position = "dodge")+
  scale_fill_brewer(palette = 'Set2')+ ggtitle("Distribution of incomelevels wrt marital status")

        ggplot(clean_train, aes(x = race, fill=incomelevel))+  geom_bar(stat="count",position = "dodge")+
  scale_fill_brewer(palette = 'Set2')+ggtitle("Distribution of incomelevels wrt race")

#ggplot(clean_train, aes(x = hoursperweek, fill=incomelevel))+ geom_bar(stat="count",position = "dodge",binwidth = 10)+
#  scale_fill_brewer(palette = 'Set1') + ggtitle("Distribution of incomelevels wrt hours per week")

#ggplot(clean_train, aes(x = age, fill=incomelevel))+ geom_bar(stat="count",position = "dodge",binwidth = 10)+
#scale_fill_brewer(palette = 'Set1')+ggtitle("Distribution of incomelevels wrt age group")

ggplot(clean_train, aes(x = workclass, fill = incomelevel)) + geom_bar(position="fill") + theme(axis.text.x = element_text(angle = 70)) + ggtitle("Workclass") + scale_fill_manual(values=c("#F8766D", "#05a815"))
ggplot(clean_train, aes(x = relationship, fill = incomelevel)) + geom_bar(position="fill") + theme(axis.text.x = element_text(angle = 70)) + ggtitle("Relationship") + scale_fill_manual(values=c("#F8766D", "#05a815"))
ggplot(clean_train, aes(x = race, fill = incomelevel)) + geom_bar(position="fill") + theme(axis.text.x = element_text(angle = 90)) + ggtitle("Race") + scale_fill_manual(values=c("#F8766D", "#05a815"))
ggplot(clean_train, aes(x = sex, fill = incomelevel)) + geom_bar(position="fill") + theme(axis.text.x = element_text(angle = 90)) + ggtitle("Gender") + scale_fill_manual(values=c("#F8766D", "#05a815"))

#--------------------------------------------- HISTOGRAMS  ------------------------------------------------------------------------------------------------------------------------------------------------------------------

p1 <- ggplot(clean_train, aes(x=age)) + ggtitle("Age") +
  geom_histogram(aes(y = 100*(..count..)/sum(..count..)), binwidth=5, colour="blue", fill="yellow") + ylab("Percentage")
p2 <- ggplot(clean_train, aes(x=log10(fnlwgt))) + ggtitle("log( Weight )") +
  geom_histogram(aes(y = 100*(..count..)/sum(..count..)), colour="blue", fill="yellow") + ylab("Percentage")
p3 <- ggplot(clean_train, aes(x=educationnum)) + ggtitle("Years of Education") + 
  geom_histogram(aes(y = 100*(..count..)/sum(..count..)), binwidth=1, colour="blue", fill="yellow") + ylab("Percentage")
p4 <- ggplot(clean_train, aes(x=hoursperweek)) + ggtitle("Hours per Week") +
  geom_histogram(aes(y = 100*(..count..)/sum(..count..)), colour="blue", fill="yellow") + ylab("Percentage")
p5 <- ggplot(clean_train, aes(x=log10(capitalgain+1))) + ggtitle("log( Capital Gain )") +
  geom_histogram(aes(y = 100*(..count..)/sum(..count..)), colour="blue", fill="yellow") + ylab("Percentage") + 
  annotate("text", x = 3, y = 50, label = "X", colour="red", size=30, fontface="bold")
p6 <- ggplot(clean_train, aes(x=log10(capitalloss+1))) + ggtitle("log( Capital Loss )") +
  geom_histogram(aes(y = 100*(..count..)/sum(..count..)), colour="blue", fill="yellow") + ylab("Percentage") + 
  annotate("text", x = 2, y = 50, label = "X", colour="red", size=30, fontface="bold")
grid.arrange(p1, p2, p3, p4, p5, p6, ncol=3)

#-------------------------------------------- BAR PLOTS -------------------------------------------------------------------------------------

p1 <- ggplot(clean_train, aes(x=workclass)) + ggtitle("Work Class") + xlab("Work Class") + geom_bar(aes(y = 100*(..count..)/sum(..count..))) + ylab("Percentage") + coord_flip() + scale_x_discrete(limits = rev(levels(workclass)))
p2 <- ggplot(clean_train, aes(x=education)) + ggtitle("Education") + xlab("Education") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..))) + ylab("Percentage") + coord_flip() +
  scale_x_discrete(limits = rev(levels(education)))
p3 <- ggplot(clean_train, aes(x=occupation)) + ggtitle("Occupation") + xlab("Occupation") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..))) + ylab("Percentage") + coord_flip() +
  scale_x_discrete(limits = rev(levels(occupation)))
p4 <- ggplot(clean_train, aes(x=nativecountry)) + ggtitle("Native Country") + xlab("Native Country") +
  geom_bar(aes(y = 100*(..count..)/sum(..count..))) + ylab("Percentage") + coord_flip() + 
  scale_x_discrete(limits = rev(levels(nativecountry))) +
  annotate("text", x = 21, y = 50, label = "X", colour="red", size=30, fontface="bold")
grid.arrange(p1, p2, p3, p4, ncol=2)

#-------------------------------------------- PIE CHARTS -----------------------------------------------------------------------------------------------

p1 <- ggplot(clean_train, aes(x=factor(1), fill=maritalstatus)) + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 1) + coord_polar(theta="y") + 
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(), legend.title=element_blank()) + 
  xlab("") + ylab("") + ggtitle("Marital Status") 
p2 <- ggplot(clean_train, aes(x=factor(1), fill=relationship)) + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 1) + coord_polar(theta="y") + 
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(), legend.title=element_blank()) + 
  xlab("") + ylab("") + ggtitle("Relationship") 
p3 <- ggplot(clean_train, aes(x=factor(1), fill=race)) + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 1) + coord_polar(theta="y") + 
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(), legend.title=element_blank()) + 
  xlab("") + ylab("") + ggtitle("Race")
p4 <- ggplot(clean_train, aes(x=factor(1), fill=sex)) + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 1) + coord_polar(theta="y") + 
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(), legend.title=element_blank()) + 
  xlab("") + ylab("") + ggtitle("Sex")+scale_fill_manual(values=c("#FFCC00", "#00BA38"))
grid.arrange(p1, p2, p3, p4, ncol=2)


#########################################  CLASSIFICATION USING NAIVE BAYES  #####################################################
#tic()
# making model for naive bayes which will be used for prediction 
model_nb = naiveBayes(incomelevel ~.,data = clean_train)

# prediction on train and test data
prediction_nb_train = predict(model_nb,train[,-15])
prediction_nb_test = predict(model_nb,test[,-15])

# confusion matrix for train and test data
conf_mat_nb_train = table(prediction_nb_train, train[,15])
conf_mat_nb_test = table(prediction_nb_test,test[,15])

# print both confusion matrix
print("Confusion matrix for train data")
print(conf_mat_nb_train)
print("confusion matrix for test data")
print(conf_mat_nb_test)

# error rate for training and testing data
error_rate_test_nb =(conf_mat_nb_test[1,2]+conf_mat_nb_test[2,1])/(conf_mat_nb_test[1,2]+conf_mat_nb_test[2,1]+conf_mat_nb_test[1,1]+conf_mat_nb_test[2,2])
print("Error rate for Naive Bayes on test data")
print(error_rate_test_nb)

error_rate_train_nb =(conf_mat_nb_train[1,2]+conf_mat_nb_train[2,1])/(conf_mat_nb_train[1,2]+conf_mat_nb_train[2,1]+conf_mat_nb_train[1,1]+conf_mat_nb_train[2,2])
print("Error rate for Naive Bayes on train data")
print(error_rate_train_nb)

#toc()
#######################################  LOGISTIC REGRESSION  ##################################################################
#tic()


library(ROCR)

glmfit = glm(income ~.,data = train,family=binomial('logit'))
summary(glmfit)
threshold <- seq(0, 1, 0.04)
acc <- rep(0, length(threshold))
for (i in 1:length(threshold)) {
  glmprobs <- predict(glmfit, newdata=test, type = "response")
  glmpred <- rep("<=50k", length(test$income))
  glmpred[glmprobs > threshold[i]] = ">50k"
  glmtable <- table(glmpred, test$income)
  acc[i] <- sum(diag(glmtable))/sum(glmtable)
}
plot(acc, main="Threshold Selection")
threshold[which.max(acc)]
log.acc <- acc[which.max(acc)]
log.acc

tb3 <- table(glmpred, test$income)
tb3
glmpred<-predict(glmtable, newdata=test, type="response")
confusionmatrix_LR<- table(test$income, glmpred > 0.5)
confusionmatrix_LR


ROCRpred<- prediction(glmprobs, test$income)
perf<- performance(ROCRpred, "tpr", "fpr")
plot(perf)
as.numeric(performance(ROCRpred, "auc")@y.values)

#accuracy 85.9


#toc()
#Baseline = Everyone makes below 50k

######################################   LOGISTIC REGRESSION EXCLUDING DIFFERENT VARIABLES #########################################

glmfit1 <- glm(income ~.-fnlwgt, data=train, family=binomial)
summary(glmfit1)
#incorporate train and set data set
#Looping through various threshold to find the best one
threshold <- seq(0, 1, 0.04)
acc <- rep(0, length(threshold))
for (i in 1:length(threshold)) {
  glmprobs <- predict(glmfit1, newdata=test, type = "response")
  glmpred <- rep("<=50k", length(test$income))
  glmpred[glmprobs > threshold[i]] = ">50k"
  glmtable <- table(glmpred, test$income)
  acc[i] <- sum(diag(glmtable))/sum(glmtable)
}
plot(acc, main="Threshold Selection")
threshold[which.max(acc)]
log.acc2 <- acc[which.max(acc)]
log.acc2

# accuracy = 85.5

#toc()

#####################################################################################
glmfit3 <- glm(income ~.-fnlwgt-nativecountry, data=train, family=binomial)
summary(glmfit3)
#incorporate train and set data set
#Looping through various threshold to find the best one
threshold <- seq(0, 1, 0.04)
acc <- rep(0, length(threshold))
for (i in 1:length(threshold)) {
  glmprobs <- predict(glmfit3, newdata=test, type = "response")
  glmpred <- rep("<=50k", length(test$income))
  glmpred[glmprobs > threshold[i]] = ">50k"
  glmtable <- table(glmpred, test$income)
  acc[i] <- sum(diag(glmtable))/sum(glmtable)
}
plot(acc, main="Threshold Selection")
threshold[which.max(acc)]
log.acc3 <- acc[which.max(acc)]
log.acc3

# accuracy = 


###################################################################################################################

#toc()

######################################   SVM  ##########################################################################################

# with library kernlab
library(kernlab)
svm4 <- ksvm(income ~ ., data = train)
svm4.pred.prob <- predict(svm4, newdata = test, type = 'decision')
svm4.pred <- predict(svm4, newdata = test, type = 'response')
confusionMatrix(test$income, svm4.pred)

# accuracy 87.22


######## Extras #########
svm.model<- svm(income~., data = train,kernel = "radial", cost = 1, gamma = 0.1)
svm.predict <- predict(svm.model, test)
svm.pred.prob <- predict(svm.model, newdata = test, type = 'decision')
confusionMatrix(test$income, svm.predict)
#Accuracy 86.72



svm.model1<- svm(income~age + workclass + education + capitalgain + occupation + nativecountry + sex + hoursperweek  + maritalstatus, data = train,kernel = "radial", cost = 1, gamma = 0.1)
svm.predict1 <- predict(svm.model1, test)
confusionMatrix(test$income, svm.predict1)
pref<-confusionMatrix(test$income, svm.predict1)
pref$table[2]
#Accuracy 85.8

## manual roc
r=pref$table[1,2]/(pref$table[1,2]+pref$table[2,2])
1-r

# roc = 0.72
#toc()





######################################   NEURAL NETWORK ###########################################################################
#tic()
library(nnet)
library(plotnet)
nn <- nnet(income ~ ., data = train, size = 40, maxit = 500)

nn.pred <- predict(nn, newdata = test, type = 'raw')

pred <- rep('<=50K', length(nn.pred))
pred[nn.pred>=.5] <- '>50K'

plot(nn.pred, y_names = "IncomeLevel")
title("Graphical Representation of our Neural Network")

# confusion matrix 
tb1 <- table(pred, test$income)
tb1
## manual roc ##
#r=tb1[1,2]/(tb1[1,2]+tb1[2,2])
#1-r
# roc =  -- 8 and 8000

emp <- data.frame( 
  tb1,
  stringsAsFactors = FALSE
)
star <- function(accuracy, precision, sensitivity, specificity) {
  a <- emp[1,3]
  b <- emp[2,3]
  c <- emp[3,3]
  d <- emp[4,3]
  accuracy <- ((a + d) / (a + b + c + d)) *100
  cat("the accuracy is ", accuracy, "%")
  precision <- ((a) / (a + c)) *100
  cat("\nthe precision is ", precision, "%")
  sensitivity <- ((a) / (a + b)) *100
  cat("\nthe sensitivity is ", sensitivity, "%")
  specificity <- ((d) / (c + d)) *100
  cat("\nthe specificity is ", specificity, "%")
}
star(accuracy, precision, sensistvity, specificity)

ROCRpred<- prediction(nn.pred, test$income)
perf<- performance(ROCRpred, "tpr", "fpr")
plot(perf)
as.numeric(performance(ROCRpred, "auc")@y.values)
#toc()

######################################  GRADIENT BOOSTING #####################################################################
#tic()
boost.fit <- gbm(as.numeric(income)-1~.,data=train, distribution =
                   "bernoulli", n.trees = 2000 , interaction.depth = 2)
summary(boost.fit)
gbm.perf(boost.fit)
threshold <- seq(0, 1, 0.04)
acc <- rep(0, length(threshold))
for (i in 1:length(threshold)) {
  boost.probs <- predict(boost.fit, test, n.trees = 2000)
  boost.pred <- ifelse(boost.probs > threshold[i], 1, 0)
  acc[i] <- mean(boost.pred == as.numeric(test$income)-1)
}
plot(acc, main="Threshold Selection")
boost.acc <- acc[which.max(acc)]
boost.acc

tb2 <- table(boost.pred, test$income)
tb2



#accuracy = 89.13 %


######################################  CLASSIFICATION USING RANDOM FOREST  ###################################################
#tic()
rf.income <- randomForest(income~.-fnlwgt, train)
#rf.pred.prob <- predict(rf, newdata = testing_set, type = 'prob')
rf.pred.prob <- predict(rf.income, newdata=test,type="prob")
rf.pred <- predict(rf.income, newdata=test)
rf.table <- table(rf.pred, test$income)
rf.acc <- mean(rf.pred == test$income)
rf.acc


#toc()
# acc = 88.64

####### Extras ###
#-------------------------------------------------------------
rf.income <- randomForest(income~.-nativecountry, train)
rf.pred <- predict(rf.income, newdata=test)
rf.table <- table(rf.pred, test$income)
rf.acc <- mean(rf.pred == test$income)
rf.acc

# acc = 88.89

tb4 <- table(rf.pred, test$income)
tb4

##############################  Classification using decison trees ##############################################################
#tic()
# model generation for decision trees
tree  = rpart(income ~ ., data = train, method ="class")
fancyRpartPlot(tree, main = "Decision tree for Adult dataset")


# predictions on test and train data
prediction_dt_test=  predict(tree, test[,-15])
prediction_dt_train=  predict(tree, train[,-15])
predictions_test = ifelse(prediction_dt_test[,1] >= .5, " <=50K", " >50K")
predictions_train = ifelse(prediction_dt_train[,1] >= .5, " <=50K", " >50K")


# print the confusion matrix for both test and train data
conf_mat_dt_train = table(predictions_train,train[,15])
conf_mat_dt_test = table(predictions_test,test[,15])

# print both confusion matrix
print("Confusion matrix for train data")
print(conf_mat_dt_train)
print("confusion matrix for test data")
print(conf_mat_dt_test)


# print error rate for both train and test data
error_rate_test_dt =(conf_mat_dt_test[1,2]+conf_mat_dt_test[2,1])/(conf_mat_dt_test[1,2]+conf_mat_dt_test[2,1]+conf_mat_dt_test[1,1]+conf_mat_dt_test[2,2])
print("Error rate for Decision Trees on test data")
print(error_rate_test_dt)

error_rate_train_dt =(conf_mat_dt_train[1,2]+conf_mat_dt_train[2,1])/(conf_mat_dt_train[1,2]+conf_mat_dt_train[2,1]+conf_mat_dt_train[1,1]+conf_mat_dt_train[2,2])
print("Error rate for Decision Trees on train data")
print(error_rate_train_dt)

#toc()
######################################## ROC VALUES #############################################################################################################

pr <- prediction(glmprobs, test$income)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")

# create a data frame for TP and FP rates
dd <- data.frame(FP = prf@x.values[[1]], TP = prf@y.values[[1]])

# NN
pr1 <- prediction(nn.pred, test$income)
prf1 <- performance(pr1, measure = "tpr", x.measure = "fpr")
dd1 <- data.frame(FP = prf1@x.values[[1]], TP = prf1@y.values[[1]])


# RF
pr2 <- prediction(rf.pred.prob[,2], test$income)
prf2 <- performance(pr2, measure = "tpr", x.measure = "fpr")
dd2 <- data.frame(FP = prf2@x.values[[1]], TP = prf2@y.values[[1]])

# SVM
pr4 <- prediction(svm4.pred.prob, test$income)
prf4 <- performance(pr4, measure = "tpr", x.measure = "fpr")
dd4 <- data.frame(FP = prf4@x.values[[1]], TP = prf4@y.values[[1]])

#GBM
pr5 <- prediction(boost.probs, test$income)
prf5 <- performance(pr5, measure = "tpr", x.measure = "fpr")
dd5 <- data.frame(FP = prf5@x.values[[1]], TP = prf5@y.values[[1]])


# plot ROC curve for logistic regression
g <- ggplot() + 
  geom_line(data = dd, aes(x = FP, y = TP, color = 'Logistic Regression')) + 
  geom_line(data = dd1, aes(x = FP, y = TP, color = 'Neural Networks')) + 
  geom_line(data = dd2, aes(x = FP, y = TP, color = 'Random Forest')) +
  geom_line(data = dd4, aes(x = FP, y = TP, color = 'Support Vector Machine')) +
  geom_line(data = dd5, aes(x = FP, y = TP, color = 'Gradient Boosting Machine')) + 
  geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1)) +
  ggtitle('ROC Curve') + 
  labs(x = 'False Positive Rate', y = 'True Positive Rate') 


g +  scale_colour_manual(name = 'Classifier', values = c('Logistic Regression'='#E69F00', 
                                                         'Neural Networks'='#56B4E9', 
                                                         'Random Forest'='#D55E00', 'Support Vector Machine'='#0072B2','Gradient Boosting Machine'='#009E73'))

auc <- rbind(performance(pr, measure = 'auc')@y.values[[1]],
             performance(pr1, measure = 'auc')@y.values[[1]],
             performance(pr2, measure = 'auc')@y.values[[1]],
             performance(pr4, measure = 'auc')@y.values[[1]],
             performance(pr5, measure = 'auc')@y.values[[1]])
rownames(auc) <- (c('Logistic Regression', 'Neural Networks', 
                    'Random Forest', 'Support Vector Machine','Gradient Boosting Machine'))
colnames(auc) <- 'Area Under ROC Curve'
round(auc, 4)

toc()

#######################################################################################
# accuracy <- round(sum(predictions == test[,15])/length(predictions), digits = 4)
# print(paste("The model correctly predicted the test outcome ", accuracy*100, "% of the time", sep=""))
# predictions <- ifelse(outcomes[,1] >= .5, " <=50K", " >50K") 
# plot(fit, uniform=TRUE,main="Classification Tree without pruning")
# text(fit, use.n=TRUE, all=TRUE, cex=.8)
# 
# 
# #pruning the tree
# pfit= prune(fit, cp=0.013441)
# 
# # plot the pruned tree 
# plot(pfit, uniform=TRUE, main="Pruned Classification Tree for g3 using class as method")
# text(pfit, use.n=TRUE, all=TRUE, cex=.8)



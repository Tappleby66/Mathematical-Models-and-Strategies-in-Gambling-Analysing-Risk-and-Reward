
library(rpart)
library(rpart.plot)

data <- data.frame(
  LovesDrinking = c("Yes", "Yes", "No", "No", "Yes", "Yes", "No"),
  LovesPoker = c("Yes", "No", "Yes", "Yes", "Yes", "No", "No"),
  Age = c(7, 12, 18, 35, 38, 50, 83),
  LovesMaths = c("No", "No", "Yes", "Yes", "Yes", "No", "No")  
)

data$LovesDrinking <- as.factor(data$LovesDrinking)
data$LovesPoker <- as.factor(data$LovesPoker)
data$LovesMaths <- as.factor(data$LovesMaths)

# Train a decision tree model
model <- rpart(LovesMaths ~ LovesDrinking + LovesPoker + Age, data = data, method = "class",control = rpart.control(maxdepth = 3, minsplit = 2, cp = 0))

# Plot the full tree
rpart.plot(model, type = 5, extra = 1, box.palette = "RdYlGn", shadow.col = "gray", 
           nn = TRUE, main = "Full Decision Tree for LovesMaths")
pdf("adjusted_full_decision_tree.pdf")
rpart.plot(model, type = 5, extra = 1, box.palette = "RdYlGn", shadow.col = "gray", 
           nn = TRUE, main = "Full Decision Tree for LovesMaths")
dev.off()

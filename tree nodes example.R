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

# Function to create and plot a decision tree with only one split
plot_root_node <- function(formula, data, title) {
  model <- rpart(formula, data = data, method = "class", control = rpart.control(maxdepth = 1, minsplit = 2, cp = 0))
  
  rpart.plot(
    model, type = 5, extra = 101, box.palette = "RdYlGn", shadow.col = "gray",
    nn = TRUE, main = title, cex.main = 1.5, tweak = 1.2
  )
}



pdf("improved_root_node_choices.pdf", width = 10, height = 4)


par(mfrow = c(1, 2), mar = c(5, 5, 4, 2))  


plot_root_node(LovesMaths ~ LovesDrinking, data, "Root Node: LovesDrinking")
plot_root_node(LovesMaths ~ LovesPoker, data, "Root Node: LovesPoker")


dev.off()

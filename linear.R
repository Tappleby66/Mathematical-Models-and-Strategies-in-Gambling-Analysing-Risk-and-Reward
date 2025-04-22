library(ggplot2)


data <- data.frame(
  Dose = c(0.00, 0.25, 0.75, 1.00, 1.50, 2.50, 3.00, 4.50),
  Response = c(1, 8, 46, 58, 102, 140, 424, 680)
)

# Manually select the 3rd and 7th data points for the testing set
test_indices <- c(3, 7)
train_indices <- setdiff(1:nrow(data), test_indices)  # All other points are training

# Split data into training and testing sets
train_set <- data[train_indices, ]
test_set <- data[test_indices, ]

# Fit a linear model to the training data
model <- lm(Response ~ Dose, data = train_set)

# Get the fitted values 
train_set$predicted <- predict(model, newdata = train_set)

# Get predictions for the testing set 
test_set$predicted <- predict(model, newdata = test_set)

# Calculate residuals for both training and testing sets
train_set$residuals <- train_set$Response - train_set$predicted
test_set$residuals <- test_set$Response - test_set$predicted

# Combine the training and testing sets for plotting
combined_data <- rbind(
  cbind(train_set, set = "Training"),
  cbind(test_set, set = "Testing")
)

# Create the plot with residuals 
plot<-ggplot(combined_data, aes(x = Dose, y = Response, color = set)) +
  geom_point(size = 3) +  
  geom_abline(intercept = coef(model)[1], slope = coef(model)[2], linetype = "solid", color = "red") + 
  geom_segment(aes(xend = Dose, yend = predicted), linetype = "dotted", color = "black") +  # Residuals as dotted lines
  theme_minimal() +
  labs(title = "Linear Model Fit with Residuals for Training and Testing Sets", 
       x = "Dose", 
       y = "Response") +
  scale_color_manual(values = c("blue", "green"))
ggsave("linear_model_residuals_plot.pdf", plot = plot, width = 6, height = 4, dpi = 300)

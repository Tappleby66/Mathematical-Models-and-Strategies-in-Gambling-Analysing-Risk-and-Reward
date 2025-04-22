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

# Fit a polynomial model 
poly_model <- lm(Response ~ poly(Dose, 5), data = train_set)

# Get the fitted values 
train_set$poly_predicted <- predict(poly_model, newdata = train_set)

# Calculate residuals 
train_set$poly_residuals <- train_set$Response - train_set$poly_predicted
test_set$poly_predicted <- predict(poly_model, newdata = test_set)
test_set$poly_residuals <- test_set$Response - test_set$poly_predicted

# Combine the training and testing sets for plotting
combined_data <- rbind(
  cbind(train_set, set = "Training"),
  cbind(test_set, set = "Testing")
)
combined_data
# Create the plot with the polynomial model and residuals
plot <- ggplot(combined_data, aes(x = Dose, y = Response, color = set)) +
  geom_point(size = 3) +  
  geom_smooth(data = train_set, aes(x = Dose, y = Response), 
              method = "lm", formula = y ~ poly(x, 5), 
              color = "purple", linetype = "dashed", se = FALSE) + 
  geom_segment(data = combined_data, aes(xend = Dose, yend = poly_predicted), linetype = "dotted", color = "black") +  
  theme_minimal() +
  labs(title = "Polynomial Model with Residuals for Training and Testing Sets", 
       x = "Dose", 
       y = "Response") +
  scale_color_manual(values = c("blue", "green"))
plot
ggsave("overfitting_poly_model_plot.pdf", plot = plot, width = 6, height = 4, dpi = 300)

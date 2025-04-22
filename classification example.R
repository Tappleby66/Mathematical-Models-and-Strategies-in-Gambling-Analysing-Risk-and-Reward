library(ggplot2)
data<-data.frame(
Effectiveness= c(2,3,4,100,100,100,50,10,2,3),
Dose = c(1,3,5,18,22,23,30,35,37,40)
)

plot<-ggplot(data, aes(x = Dose, y = Effectiveness)) +
  geom_point(size = 3 , colour = "red")+
  theme_minimal() 
ggsave("drug_plot.pdf", plot = plot, width = 6, height = 4, dpi = 300)



data2 <- data.frame(
  Effectiveness = c(4, 100, 50, 10),
  Effective =c(0,1,0,0),
  Dose = c(5, 18, 30, 35)
)

# Initial prediction 
initial_prediction <- 0.5

# Add residual
data2$residual <- data2$Effective - initial_prediction

# Midpoints
split_points <- c(11.5, 24, 32.5)

# Plot with prediction line, residuals, and split lines
plot2 <- ggplot(data2, aes(x = Dose, y = Effective)) +
  geom_point(size = 5, colour = "blue") +
  geom_hline(yintercept = initial_prediction, linetype = "dashed", color = "red") +
  geom_segment(aes(xend = Dose, yend = initial_prediction), linetype = "dotted", color = "black") +
  geom_vline(xintercept = split_points, linetype = "dotdash", color = "darkgreen") +
  theme_minimal()

plot2


ggsave("drug_plot3.pdf", plot = plot2, width = 6, height = 4, dpi = 300)


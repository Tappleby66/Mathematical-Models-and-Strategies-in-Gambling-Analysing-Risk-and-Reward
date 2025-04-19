#install.packages("ggplot2")
#install.packages("dplyr")
#install.packages("purrr")
#install.packages("e1071")
#install.packages("pastecs")
#install.packages("psych")
#install.packages("MASS" )
install.packages("gridExtra")
install.packages("tidyr")
install.packages("Matrix")
library(MASS)
library(ggplot2)
library(dplyr)
library(purrr)
library(e1071)  
library(pastecs)
library(psych)
library(castor)
library(gridExtra)
library(tidyr)
#Simulates 1 spin of a roulette wheel
roulette <- function() {
  a <- sample(0:37, 1)
  if (a > 19) return(1) else return(0)
}
# The different betting strategies
dalembert <- function(sims, cash, bet, goal) {
  total <- c(cash)
  for (i in 1:sims) {
    win <- roulette()
    if (win > 0) {
      cash <- cash + bet
      if (cash >= goal) break
      bet <- max(bet - 1, 1)
    } else {
      cash <- cash - bet
      if (cash <= 0) break
      if (cash>=bet+1){
        bet<-bet+1
      }
      else {bet<-cash}
    }
    total <- c(total, cash)
  }
  
  if (cash <= 0) {
    total <- c(total, rep(0, sims + 1 - length(total)))
  } else {
    total <- c(total, rep(goal, sims + 1 - length(total)))
  }
  return(total)
}

martingale <- function(sims, cash, bet, goal) {
  total <- c(cash)
  for (i in 1:sims) {
    win <- roulette()
    if (win > 0) {
      cash <- cash + bet
      bet <- 1
      if (cash >= goal) break
    } else {
      cash <- cash - bet
      if (cash <= 0) break
      if (cash>=bet*2){
        bet<-bet*2
      }
      else {bet<-cash}
    }
    total <- c(total, cash)
  }
  
  if (cash <= 0) {
    total <- c(total, rep(0, sims + 1 - length(total)))
  } else {
    total <- c(total, rep(goal, sims + 1 - length(total)))
  }
  return(total)
}
revmartingale <- function(sims, cash, bet, goal) {
  total <- c(cash)
  for (i in 1:sims) {
    win <- roulette()
    if (win > 0) {
      cash <- cash + bet
      if (cash >= goal) break
      bet <- bet * 2
    } else {
      cash <- cash - bet
      if (cash <= 0) break
      bet <- max(bet / 2, 1)
    }
    total <- c(total, cash)
  }
  
  if (cash <= 0) {
    total <- c(total, rep(0, sims + 1 - length(total)))
  } else {
    total <- c(total, rep(goal, sims + 1 - length(total)))
  }
  return(total)
}


#Simulates multiple players using the same betting strategy
rungame <- function(strat, sims, cash, bet, goal,plots) {
  fails <- 0
  mydata <- numeric()
  for (run in 1:plots) {
    results <- strat(sims, cash, bet, goal)
    if (tail(results, 1) == 0) {
      fails <- fails + 1
    }
    mydata <- c(mydata, tail(results, 1))
  }
  print(fails/plots*100)
  print(describe(mydata))
  barmydata<-sort(mydata, decreasing = FALSE)
  hist(barmydata, main="Histogram of Final Balance with D'Alembert", ylab="Frequency",xlab="Balance", col="skyblue",cex.main=0.8)
}


rungame(martingale, 200, 100, 2, 200, 1000)
rungame(dalembert,200,100,1,200,10000)
# Plots the graphs of player balance using a betting strategy 
plotresult1<-function(strat,sims,cash,bet,goal,plots){
  i=1
  iterations<-1:(sims+1)
  plot_data <- data.frame(iterations)
  for (i in 1:plots) {
    results <- strat(sims,cash,bet,goal)
    plot_data<-cbind(plot_data,results)
    colnames(plot_data)[ncol(plot_data)] <- as.character(i)
    i=i+1
    
  }
  df_long <- plot_data %>%
    pivot_longer(
      cols = -iterations,     
      names_to = "results",   
      values_to = "value"    
    )
  pdf("martingale_article.pdf", width = 5, height = 3)
  p<-ggplot(df_long, aes(x = iterations, y = value, color = results)) +
    geom_line(size = 1) +
    labs(
      title = "Martingale Strategy",
      x = "Iteration",
      y = "Balance"
    ) +
    theme_minimal()+
    theme(legend.position = "none")
  print(p)
  dev.off()
}


plotresult1(martingale, 200, 100, 1, 200, 5)

#Probability of not going bankrupt for each gain of unit(£1) starting with £500
y<-pgeom(7,18/37,lower.tail = FALSE)
#So to gain £100
pgain100<-y**100
#So the probability of going Bankrupt to gain £100 is :
pbankrupt<-1-pgain100
pbankrupt
#which is close to simulated bankruptcy percentage 
#Matrix for small markov martingale problem 
W<-18/37
L<-19/37
markov1<-matrix(c(1,0,0,0,0,0,0,0,0,0,
                  L,0,W,0,0,0,0,0,0,0,
                  0,L,0,0,W,0,0,0,0,0,
                  L,0,0,0,0,0,W,0,0,0,
                  0,0,L,0,0,0,W,0,0,0,
                  0,L,0,0,0,0,0,0,W,0,
                  0,0,0,0,0,L,0,0,W,0,
                  0,0,0,L,0,0,0,0,0,W,
                  0,0,0,0,0,0,0,L,0,W,
                  0,0,0,0,0,0,0,0,0,1), 
                nrow = 10 , ncol = 10, byrow= TRUE)
markov2<-matrix(c(1,0,0,0,0,0,
                  0.5,0,0.5,0,0,0,
                  0,0.5,0,0,0.5,0,
                  0.5,0,0,0,0,0.5,
                  0,0,0,0.5,0,0.5,
                  0,0,0,0,0,1),
                nrow = 6, ncol = 6, byrow = TRUE )
markov3<-matrix(c(1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  L,0,0,0,W,0,0,0,0,0,0,0,0,0,0,0,
                  L,0,0,0,0,0,0,0,0,0,W,0,0,0,0,0,
                  L,0,0,0,0,0,0,0,0,0,0,W,0,0,0,0,
                  0,L,0,0,0,0,0,W,0,0,0,0,0,0,0,0,
                  0,L,0,0,0,0,0,0,0,0,0,0,W,0,0,0,
                  0,L,0,0,0,0,0,0,0,0,0,0,0,0,W,0,
                  0,0,L,0,0,0,0,0,0,0,W,0,0,0,0,0,
                  0,0,L,0,0,0,0,0,0,0,0,0,0,W,0,0,
                  0,0,0,L,0,0,0,0,0,0,0,0,0,0,W,0,
                  0,0,0,0,0,L,0,0,0,0,0,0,W,0,0,0,
                  0,0,0,0,0,0,L,0,0,0,0,0,0,0,0,W,
                  0,0,0,0,0,0,0,0,L,0,0,0,0,W,0,0,
                  0,0,0,0,0,0,0,0,0,L,0,0,0,0,W,0,
                  0,0,0,0,0,0,0,0,0,0,0,L,0,0,0,W,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1), nrow = 16 , byrow = TRUE)

#Code that builds the larger matrices 
matrix.builder<-function(endvalue){
  # First works out the size of the matrix
  matlength<-(endvalue*(endvalue-1))/2+2
  mat<-matrix(0,nrow = matlength,ncol = matlength)
  W<-18/37
  L<-19/37
  finaloutput<-c()
  # The stop seq adds positions to the loss sequence that could not be captured
  stopseq<-function(len){
    stopseq<-c(7)
    for (n in 1:len) {
      stopseq<-append(stopseq,stopseq[length(stopseq)]+(n+3))
    }
    return(stopseq)
  }
  #This adds the positions of losses after an end value of 5
  listbuildL<-function(endvalue){
    listL<-list(i=c(2,4,3,4,5,6,7,8,9,10,11),
                j=c(1,1,2,2,4,2,2,6,4,2,1))
    if (endvalue>5){
      for (n in 6:endvalue){
        # Makes initial list 
        listL$i <- c(listL$i, seq(listL$i[length(listL$i)] + 1,by=1,length.out=n-1))
        listL$j <- c(listL$j, tail(listL$j, n-2))
        for (m in 3:(n-3)){
          stoplist<-stopseq(endvalue)
          # Checks to add un captured loss from first list 
          if (!(listL$j[length(listL$j) -(n-m)] %in% stoplist)){
            listL$j[length(listL$j) -(n-m)] <- listL$j[length(listL$j) - (n-m)] + 1
          }
        }
        position <- length(listL$j) - (n-2)
        value_to_insert <- listL$j[position+1] + (n-4)
        listL$j <- append(listL$j, values = value_to_insert, after = position)
      }
    }
    return(listL)
  }
  # Builds a list for the win positions 
  listbuildW<-function(endvalue){
    listW<-list(i=c(2,3,4,5),
                j = c(3,5,8,8))
    if (endvalue>5){
      for (n in 6:endvalue){
        listW$j<-append(listW$j,rep((listW$j[length(listW$j)]+(n-2)),times =(ceiling(n/2)-1) ))
        if (!(n %% 2 == 0)) {
          listW$i<-append(listW$i,listW$i[length(listW$i)-(floor(n/2)-2)]+1)}
        else{
          listW$i<-append(listW$i,listW$i[length(listW$i)-((n/2)-2)]+((n/2)-1))}
        
        for (m in 1:(ceiling(n/2)-2)){
          listW$i<-append(listW$i,listW$i[length(listW$i)]+(floor(n/2))-2+m)
        }
        if (n>6){
          for (k in 1:(ceiling(n/2)-3)){
            listW$j[length(listW$j)-(1+k)]<-listW$j[length(listW$j)-(1+k)]+k
          }
        }
      }
    }
    return(listW)
  }
  # for the small matrix cases
  if (endvalue==3){
    listL<-list(i=c(2,4,3),
                j = c(1,1,2))
    listW<-list(i=c(2,3,4),
                j = c(3,5,5))
  }
  if (endvalue==4){
    listL<-list(i=c(2,4,3,6,7,5),
                j = c(1,1,2,2,2,4))
    listW<-list(i=c(2,3,4,5,6,7),
                j = c(3,5,8,8,8,8))
  }
  if (endvalue>4){
    listL<-listbuildL(endvalue)
    listW<-listbuildW(endvalue)
  }
  # This uses the lists made to build the matirx
    mat[1,1]<-1
    mat[matlength,matlength]<-1
      for (i in 1:matlength){
        for (j in 1:matlength){
          if (any(i == listL$i & j == listL$j)){
            mat[i,j]<-L
          }
          if (any(i == listW$i & j == listW$j)){
            mat[i,j]<-W
          }
        }
          row_sum <- sum(mat[i, ])
          # Adds all the wins that hit end value 
          if (row_sum < 1) {
            mat[i, ncol(mat)] <- (1 - row_sum)
          }
        
      }
    
return(mat)
}

# Code to add labels for plotting the matrices
generate_state_labels <- function(num_states) {
  
  state_labels <- vector("character", num_states)
  first_coord<-0
  second_coord<-0
  
  for (i in 0:(num_states-1)) {
    state_labels[i + 1] <- paste("(", first_coord, ",", second_coord, ")", sep = "")
    if (first_coord == second_coord){
      first_coord <- first_coord+1
      second_coord <-1
    }
    else {
        second_coord <- second_coord +1 
    }
  }
  
  return(state_labels)
}


# Plots the matrices
plot_transition_matrix_wl <- function(mat, filename = "matrix_wl_plot.pdf", title = "Transition Matrix ") {
  mat_dense <- as.matrix(mat)
  
  num_states <- nrow(mat_dense)
  state_labels <- generate_state_labels(num_states)
  
  df <- as.data.frame(as.table(mat_dense))
  colnames(df) <- c("From", "To", "Probability")
  
  W_prob <- 18/37
  L_prob <- 19/37
  
  df$Type[abs(df$Probability - W_prob) < 1e-6] <- "Win"
  df$Type[abs(df$Probability - L_prob) < 1e-6] <- "Loss"
  df$Type[df$Probability == 1 & df$From == df$To] <- "Absorbing"
  print(df)

  type_colors <- c("Win" = "forestgreen", "Loss" = "firebrick", "Absorbing" = "steelblue")
  
  
  p <- ggplot(df, aes(x = To, y = From, fill = Type)) +
    geom_tile(color = "white") +
    scale_fill_manual(values = type_colors, name = "Transition Type") +
    labs(title = title, x = "To State", y = "From State") +
    theme_minimal() +
    theme(
      axis.text.x = element_blank(),
      axis.text.y = element_blank(),
      plot.title = element_text(size = 10, face = "bold")
    )
  
  
  ggsave(filename, plot = p, width = 6, height = 5)
  
}
# Monte Carlo Approximation
simulate_absorption <- function(start_state,markovchain,lastcol, n_sim = 10000, max_steps = 1000) {
  absorbing_states <- c(1, lastcol)
  absorption_counts <- c(0, 0)
  
  for (i in 1:n_sim) {
    state <- start_state
    steps <- 0
    while (!(state %in% absorbing_states) && steps < max_steps) {
      probs <- markovchain[state, ]
      state <- sample(1:lastcol, size = 1, prob = probs)
      steps <- steps + 1
    }
    if (state == 1) absorption_counts[1] <- absorption_counts[1] + 1
    if (state == lastcol) absorption_counts[2] <- absorption_counts[2] + 1
  }
  
  absorption_counts / n_sim
}


z<-matrix.builder(12)
m<-nrow(z)

## Plots the probability of ending up in an absorbing state 
results <- sapply(2:(m-1), function(start_state) {
  simulate_absorption(start_state, markovchain = z, lastcol = m)
})

state_labels<- generate_state_labels(m)
full_probabilities <- c(1, as.numeric(results[1, ]), 0)

df_stationary <- data.frame(
  State = state_labels,
  Probability = full_probabilities
)

df_stationary$State <- factor(df_stationary$State, levels = df_stationary$State)
ggplot(df_stationary, aes(x = State, y = Probability)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black") +
  labs(title = "Stationary Distribution Probability of going Bankrupt", x = "State", y = "Probability") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 6),
    plot.title = element_text(size = 12, face = "bold")
  )
ggsave("bankruptcy_plot.pdf", width = 8, height = 5)


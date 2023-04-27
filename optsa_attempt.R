library(keras)
library(tidyverse)
library(tensorflow)
library(reticulate)
install_tensorflow(extra_packages = "pillow")
install_keras()
#install.packages('optimization')
#install.packages('genalg')
#library(genalg)
library(optimization)
setwd("C:/Users/ept20/OneDrive - purdue.edu/Purdue/Semester 6/IE 332/project2")

# Load a pre-trained image classification model
model <- load_model_tf("./dandelion_model")

# Load an example image and its true label
img <- image_load("./dandelions/dandelion_yellow_flower_230715-1599547728.jpg")
true_label <- "y"

#Gaussian perturbation function
apply_perturbation <- function(img, x) {
  # Reshape x to match dimensions of image
  x <- matrix(x, nrow = nrow(img), ncol = ncol(img), byrow = TRUE)
  
  # Add perturbation to image
  img_perturbed <- img + x
  
  # Clip pixel values to [0, 1] range
  img_perturbed[img_perturbed < 0] <- 0
  img_perturbed[img_perturbed > 1] <- 1
  
  # Return perturbed image
  img_perturbed
}

# Convert the image to a numeric vector
x0 <- image_to_array(img)

# Define the objective function
obj_fn <- function(x) {
  # Convert the numeric vector back to an image
  img <- array_to_img(x, dim_ordering = "tf")
  # Get the predicted class for the image
  class <- predict(model, image_to_matrix(img))
  # Return the negative probability of the target class
  -class[target_class + 1]
}

# Set the search space bounds
lower <- rep(0, length(x0))
upper <- rep(255, length(x0))

# Set the optimization control parameters
control <- list(maxit = 5000, trace = TRUE)

# Set the temperature and other parameters for the SA algorithm
temp <- 1.0
temp_min <- 0.01
alpha <- 0.99

# Run the SA optimization algorithm
result <- optim_sa(start = x0, fun = obj_fn, maximization = FALSE,
                   lower = lower, upper = upper, control = control)
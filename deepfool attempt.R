library(keras)
library(tensorflow)
library(reticulate)
library(tidyverse)
install_tensorflow(extra_packages = "pillow")
install_keras()
library("Adverserial")

# Define the DeepFool attack
attack <- deepfool(model, max_iter = 100)

# Generate an adversarial example
adv_img <- attack(img)
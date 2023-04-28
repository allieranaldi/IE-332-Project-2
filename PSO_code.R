# Written by Arisa Kulkarni

library(keras)

# Load the image classification model
model <- load_model_hdf5("./danelion_model")

# Load the test image
test_image  <- image_load(paste("./data-for-332/data-for-332/grass",i,sep=""),
                          target_size = c(224,224))

# Convert the test image to an array and normalize
x <- image_to_array(test_image)
x <- array_reshape(x, c(1, dim(x)))
x <- x / 255

# Define the parameters for the particle swarm optimization
n_particles <- 50
max_iterations <- 100
inertia_weight <- 0.7
cognitive_coeff <- 1.4
social_coeff <- 1.4
epsilon <- 0.05

# This is where we define the fitness function
fitness_function <- function(particle) {
  # For generating the adversarial image
  adv_x <- x
  adv_x[,,particle == 1] <- adv_x[,,particle == 1] + epsilon
  adv_x[adv_x > 1] <- 1
  adv_x[adv_x < 0] <- 0
  
  # For evaluating the model prediction on the adversarial image
  pred <- model %>% predict(adv_x)
  
  # This returns 1 - predicted probability of true class as fitness score
  1 - pred[1, class_idx]
}

# Get index of the test image
true_class <- which.max(model %>% predict(x))

# Randomly initialize particles and velocities 
n_pixels <- dim(x)[2] * dim(x)[3]
particles <- matrix(sample(c(0, 1), n_particles * n_pixels, replace = TRUE), ncol = n_pixels)
velocities <- matrix(runif(n_particles * n_pixels, -0.5, 0.5), ncol = n_pixels)

# initialize personal and global position and fitness
personal_best_position <- particles
personal_best_fitness <- apply(particles, 1, fitness_function)
global_best_position <- personal_best_position[which.max(personal_best_fitness),]
global_best_fitness <- max(personal_best_fitness)

# Define the sigmoid function
sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

# Define velocity and position update rules
for (iter in 1:max_iterations) {
  for (i in 1:n_particles) {
    # This updates the velocity of the particle
    cognitive_component <- cognitive_coeff * runif(n_pixels) * (personal_best_position[i,] - particles[i,])
    social_component <- social_coeff * runif(n_pixels) * (global_best_position - particles[i,])
    velocities[i,] <- inertia_weight * velocities[i,] + cognitive_component + social_component
    
    # This updates the position of the particle
    particles[i,] <- ifelse(runif(n_pixels) < sigmoid(velocities[i,]), 1, 0)
    
    # This evaluates the fitness of the particle
    fitness <- fitness_function(particles[i,])
    
    # Update personal and global best positions
    if (fitness > personal_best_fitness[i]) {
      personal_best_position[i,] <- particles[i,]
      personal_best_fitness[i] <- fitness
    }
    if (fitness > global_best_fitness) {
      global_best_position <- particles[i,]
      global_best_fitness <- fitness
    }
  }
}

# Generate the adversarial image with best particle
best_particle <- global_best_position
adv_x <- x
adv_x[,,best_particle == 1] <- adv_x[,,best_particle == 1] + epsilon
adv_x[adv_x > 1] <- 1

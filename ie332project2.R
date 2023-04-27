install.packages("tidyverse")
install.packages("keras")
install.packages("tensorflow")
install.packages("reticulate")

library(tidyverse)
library(keras)
library(tensorflow)
library(reticulate)
install_tensorflow(extra_packages="pillow")
install_keras()
setwd("Desktop/ie332project")
model<-load_model_tf("./dandelion_model")

res=c("","")
f <- list.files("./data-for-332/data-for-332/grass")
for (i in f) {
  test_image <- image_load(paste("./data-for-332/data-for-332/grass/",i,sep=""),target_size = c(224,224))
  x <- image_to_array(test_image)
  x <- array_reshape(x, c(1, dim(x)))
  x <- x/255
  pred <- model %>% predict(x)
  if(pred[1,2]<0.50){
    res <- c(res,i)
    print(i)
  }
}
print(res)

res=c("","")
f <- list.files("./data-for-332/data-for-332/dandelions")
for (i in f){
  test_image <- image_load(paste("./data-for-332/data-for-332/dandelions/",i,sep=""),target_size = c(224,224))
  x <- image_to_array(test_image)
  x <- array_reshape(x, c(1, dim(x)))
  x <- x/255
  pred <- model %>% predict(x)
  if(pred[1,1]<0.50){
    res <- c(res,i)
    print(i)
  }
}
print(res)
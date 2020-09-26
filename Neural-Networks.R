#Hi!!! Welcome to the doggo tutorial
#We'll train you on the basics of neural networks!!
#They're called like that because we try our best to mimic how the brain works
# Load the data!!!
#I'm on Windows, so remember to use the slash instead of the back slash!!!
dog_data <- read.csv("C:/Users/ADMIN1/Desktop/Projects/R/NN_to_NLP/dog_data.csv")

#First, we need to prepare the data
# Check the structure, let's make sure everything's there
str(dog_data)
head(dog_data)
summary(dog_data)

#We have 3 breeds of doggos.
#I shall call 0 the Siberian Husky pun dog
#I shall call 1 the Shiba Inu judging face
#I shall call 2 exasparated white wolf

#Split the data into the training and test data
#We also need a way to identify the data, so we create labels
#The other 3 (age, weight, height) are the features of the data
#We also need to separate the labels from the data itself
#You dont give a child the answers when you want to teach them
#Training data! And its labels
train_X <- as.matrix(dog_data[1:160, 1:3])
raw_train_Y <- as.matrix(dog_data[1:160, 4])

#test data! and its label
test_X <- as.matrix(dog_data[161:200, 1:3]) 
raw_test_Y <- as.matrix(dog_data[161:200, 4])
###

# Check first few lines of new variables to see if the output is what we expect
# Training data
head(train_X)
head(raw_train_Y)

# Test data
head(test_X)
head(raw_test_Y)

#We need to change the integer data to categorical data.
#Why? Keeping it as is introduces a bias into the model
#The model will think that the integers imply a hierachcy or ranking
#And acts on that assumption and that messes everything up
train_Y <- to_categorical(raw_train_Y, num_classes = 3)
test_Y <- to_categorical(raw_test_Y, num_classes = 3)

# Print out some ofthe training and test data to check whether a miracle happened
head(train_Y)
head(test_Y)

#A miracle happened!!
#So let's build our model!!!
#Are you using a GPU?
#If yes, uncomment and run the code below to set the number of parallel pools (or whatever, I don't know)
#If no, ignore this and move on
#use_session_with_seed(5)
#set.seed(5)

model <- keras_model_sequential()

model %>%
  
# Add densely-connected neural network layers using `layer_dense` function
# Our first layer has an input shape of 3 to represent 3 input features
  
  layer_dense(units = 10, activation = "relu", input_shape = 3) %>% 

# We now have a hidden layer with 10 nodes, with an input shape of 3 representing our 3 features.
  
# Next up we'll add another layer, with 10 nodes too.
  layer_dense(units = 10, activation = "relu") %>% 

#I feel like having three hidden layers, so :shrugs:
  layer_dense(units = 10, activation = "relu") %>% 

  layer_dense(units = 3, activation = "softmax")
#The last layer is a softmax one! 
#This is a type of logistic regression done for situations with multiple cases
#The output layer has 3 nodes, one for each type of category we have

model %>% summary

#Now, let's compile the model
#The loss function is used to tell the computer how off it is
#The optmimizer is used to guess the relation between the features
#There's also metrics, which specifies which information it provides so we can analyze the model
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adagrad(),
  metrics = c("accuracy")
)

#Ta-dah!! The model's compiled! Now let's see how it performs throughout the training set
#The epochs are the number of times we're running this
#I'd personally use 100 minimum, but this is a live session, so :shrugs:
history <- model %>% fit(
  x = train_X,
  y = train_Y,
  shuffle = T,
  epochs = 25,
  batch_size = 2,
  validation_split = 0.2
)

plot(history)

#Now run it!!
history

#Let's see how our baby model works!!!
#We need to ensure that the model is not underfitted, or overfitted
#We need to be picky like Goldilocks
#Not too hot, not too cold, it needs to be just right
perf <- model %>% evaluate(test_X, test_Y)
print(perf)

#How's the accuracy looking?
#If you don't like it, you can increase or decrease the epochs
#Now for the final test: Completely new data that we're coming up with right now
new_dog <- data.frame(age = 5, weight = 4, height = 8)

#Behold, a new doggo :slow_Clap:
str(dog_data)

#Let's see the relationships between age, height/weight, and breed
# Age vs weight
ggplot() +
  geom_point(data = dog_data, aes(x = age, y = weight, colour = as.factor(breed))) +
  geom_point(data = new_dog, aes (x = age, y = weight), shape = "+", size=10) +
  labs(x = "Age", y = "Weight", colour = "Breed")

#Age vs height
ggplot() +
  geom_point(data = dog_data, aes(x = age, y = height, colour = as.factor(breed))) +
  geom_point(data = new_dog, aes (x = age, y = height), shape = "+", size=10) + 
  labs(x = "Age", y = "Height", colour = "Breed")

#Now, let's see what our model thinks!!
print("Probabilities of classes:")
predict_proba(model, as.matrix(new_dog))

print("Predicted class:")
predict_classes(model, as.matrix(new_dog))
#So, what meme  is our doggo?
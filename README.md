# PClub Secy Task 7 - Clickbait Generator

The task to create a clickbait generator model is accomplished by using PyTorch library. I have used a LSTM followed by a hidden layer with ReLU activation function. The trained model is saved to be used for querying later. Flask is used to host the model locally on a website with a very basic UI, with a single button and textbox to show the results.

Many different modifications were tried to improve the model's predictions like - 
- modifying learning rate and number of epochs to prevent overfitting
- training the model only on a fraction of the given dataset, which improved the predictions quite significantly
- changing the length of the sequence used for training the LSTM
- varying number of stacking in LSTM layers

The generated clickbait statements are not very sensible, but this is what I could manage in a short time frame.

## Code Explanation
```model.py``` contains the code for the LSTM model class. 

```dataset.py``` is used to read the data from csv file and convert it to ```torch.utils.data.Dataset``` object. The ```__get_item__``` method is used to generate    ```(input, target)``` pairs for the model training, eg- input is "the car crashed into", target is "crashed into the truck". 

```train.py``` is used to train the model in a mini-batch gradient descent and using Adam optimizer, with cross entropy loss. The model is then saved to the disk for later use by the API

```app.py``` contains 2 pieces of code - first to create a basic flask application which handles the POST request on the HTML website hosted locally. When it receives a post request, it generates a random clickbait title using the ```generate``` function defined.


## References 
https://www.tensorflow.org/text/tutorials/text_generation

https://closeheat.com/blog/pytorch-lstm-text-generation-tutorial

https://github.com/nikhilbarhate99/Char-RNN-PyTorch/tree/master

# VectorWolf
**VectorWolf is (Tensor => Vector) + (reverse(Flow) => Wolf)**

Syntax is almost the same as TensorFlow. 

Faster than TensorFlow for smaller batch sizes.

| Time Comparison (in sec) | VectorWolf | TensorFlow |
|----------|----------|----------|
| House Price Prediction  | 5.259  | 85.520  |
| Coffee Roasting  | 1.238  | 47.201  |

## How to use
- Write your code in `main.cpp`
- main.cpp is given as an example to demonstrate its usage.
- Ensure that relative locations of header (.cpp) files are located correctly w.r.t `main.cpp`.
- This is meant to be used accompanying with python because visualisation and feature engineering is much easier on python.
- Output processed data to .csv files and read them separately with VectorWolf.
- Train the model in VectorWolf and then again write required data into .csv file then read it again with python for further checks.

## Important Points
- To change data type for calculation go to basic.h and change - ```using D = double``` to desired data_type.
- Running on Windows Powershell is faster than running on WSL + Ubuntu. Running on DevCpp is much much faster (10x) than running on other terminals.
- **.cpp** files are treated as header instead of **.h** files to get a perfomance boost (5x).
- The dimensions of weight matrix is transpose of what is used on TensorFlow.
- Go [here](https://github.com/soham-c04/VectorWolf/tree/main/Images) for images about Running and Output.
- Losses in BinaryCrossentropy printed using VectorWolf are generally lower than TensorFlow even for the same set of weights and biases, because of inaccurate calculations due to capping at eps = 1e-15 (to prevent runtime errors). For e.g. this results in $-log(e^{-18})$ to become $-log(e^{-15})$, hence value changes from 18 to 15.

## Methods for Layer -
### 1. layers.Dense()
  **returns** -> `class Layer`
  
  **Use**: Creating a layer.
  
  **Parameters**:-
  - `units`: No. of units in that layer (necessary).
  - `activation`: Activation function for that layer (set to **linear** by default) (optional).
  - `name`: Name of layer (set to **Layer 'layer_number'** by default) (optional).

### 2. `operator()`
  **returns** -> `vector<vector<D>>`
  
  **Use**: Getting the output Matrix from layer, after giving input. 
  
  **Parameters**:-
  - `x`: Input for the layer (necessary).
  - `z_store`: If **true**, stores the Matrix z, i.e. just before applying activation function on the Matrix. (set to     **false** by default) (optional).

### 3. layer.info()
  **returns** -> `int`
  
  **Use**: Prints layer name, type, units, parameters for that layer. Returns the parameters in that layer.
  
  **Parameters**:-
  - `prev_units`: No. of units in previous layer (necessary).

### 4. layer.get_name()
  **returns** -> `string`
  
  **Use**: Getting the name of layer. 
  
  **Parameters**: None

### 5. layer.set_name()
  **returns** -> `void`
  
  **Use**: Setting/Changing the name of layer. 
  
  **Parameters**:-
  - `name__`: Updated name of the layer (necessary).

### 6. layer.get_weights()
  **returns** -> `vector<vector<D>>`
  
  **Use**: Getting the weight Matrix of layer. 
  
  **Parameters**: None

### 7. layer.set_weights()
  **returns** -> `void`
  
  **Use**: Load a weight matrix onto the layer.
  
  **Parameters**:-
  - `new_weight`: Updated weight matrix of the layer (necessary).

### 8. layer.get_bias()
  **returns** -> `vector<D>`
  
  **Use**: Getting the bias vector of layer. 
  
  **Parameters**: None

### 9. layer.set_bias()
  **returns** -> `void`
  
  **Use**: Load a bias vector onto the layer. 
  
  **Parameters**:-
  - `new_bias`: Updated bias vector of the layer (necessary).

### 10. layer.activation_name
  **returns** -> `string &`
  
  **Use**: Getting the name of activation function of the layer.
  
  **Parameters**: None
  
### 11. print(layer)
  **returns** -> `void`
  
  **Use**: Prints layer_name, units, all weights and biases for that layer.
  
  **Parameters**:-
  - `layer`: class Layer to be printed (necessary).

## Methods for Model -
### 1. keras.Sequential();
  **returns** -> `class Model`
  
  **Use**: Creating a model.
  
  **Parameters**:-
  - `input_param`: No. of features in input of training data (necessary).
  - `vector<Layer>`: Vector of layers inputed in the form ```{ layers.Dense(units = ..., activation = "...", name = "..."), ... }``` (necessary).

### 2. model.add();
  **returns** -> `void`
  
  **Use**: Adding a new layer to the model.
  
  **Parameters**:-
  - `new_layer`: New layer to be added to be model (necessary).

### 3. model.summary()
  **returns** -> `void`.
  
  **Use**: Prints the layer_name, output shape and parameters for each layer in model
  
  **Parameters**: None

### 4. model.compile()
  **returns** -> `void`.
  
  **Use**: Setting the loss function and learning_rate for the model.
  
  **Parameters**:-
  - `loss`: Loss function used in model (set to **MeanSquaredError** by default) (optional).
  - `learning_rate`: learning_rate for weights/bias (set to **0** i.e. no updates by default) (optional).

### 5. model.fit()
  **returns** -> `void`.
  
  **Use**: Running epochs (gradient descent). Prints the current_epoch number, Loss for that epoch and Time to run that epoch.
  
  **Parameters**:-
  - `x_train`: Input for training data (necessary).
  - `y_train`: Ouput for given data (necessary).
  - `epochs`: No. of epochs (set to **0** i.e. nothing happens by default) (optional).
  - `batch_size`: Input data divided into subsets (set to **32** by default) (If **$\text{batch\\_size} \nmid \text{|x\\_train|}$**, it warps around from start to compensate) (optional).
  - `steps_per_epoch`: No. of steps per epoch (set to **$\left\lceil \frac{|{x\\_train}|}{batch\\_size} \right\rceil$** by default) (optional).
  - `Shuffle`: Training data will be randomly shuffled before each epoch (set to **true** by default) (optional).

### 6. model.predict();
  **returns** -> `vector<D>`
  
  **Use**: Predicting the output from input Matrix based on the pre-compiled and fitted model. Returns output vector.
  
  **Parameters**:-
  - `x`: Input to be predicted/processed (necessary).

### 7. model.get_layer();
  **returns** -> `class Layer &`
  
  **Use**: Getting a layer by reference to do changes to it or get information.
  
  **Parameters**:-
  - `name_`: Name of layer to be returned (necessary).

### 8. model.get_layers();
  **returns** -> `vector<class Layer>`
  
  **Use**: Getting all the layers in a model.
  
  **Parameters**: None
  
### 9. print(model)
  **returns** -> `void`.
  
  **Use**: Prints [print(layer[i])](https://github.com/soham-c04/VectorWolf/tree/main?tab=readme-ov-file#printlayer) for all layers.
  
  **Parameters**:-
  - `model`: class Model to be printed (necessary).

## Other Methods
### 1. read_csv();
  **returns** -> `vector<vector<D>>`
  
  **Use**: Reads input/ouput data from a .csv file and returns as a 2D vector.
  
  **Parameters**:-
  -  `path`: Absolute/relative path to the .csv file to be read (necessary).

### 2. shape();
  **returns** -> `void`
  
  **Use**: Printing the shape of a Matrix.
  
  **Parameters**:-
  -  `M`: Matrix whose shape is to be printed (necessary).

### 3. transpose();
  **returns** -> `vector<vector<D>>`
  
  **Use**: Getting the transpose of a Matrix.
  
  **Parameters**:-
  -  `M`: Matrix to be transposed (necessary).

### 4. multiply();
  **returns** -> `vector<vector<D>>`
  
  **Use**: Multiplying two Matrices (if their dimensions match).
  
  **Parameters**:-
  -  `a`: First Matrix (necessary).
  -  `b`: Second Matrix (necessary).

### 5. print();
  **returns** -> `void`
  
  **Use**: Printing contents of a 1D vector, 2D vector, Layer or Model.
  
  **Parameters**:-
  -  `vec` or `mat` or `layer` or `model`: Any one of the four (necessary).



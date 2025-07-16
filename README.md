# VectorWolf
**VectorWolf is (Tensor => Vector) + (reverse(Flow) => Wolf)**

Syntax is almost the same as TensorFlow. 

Faster than TensorFlow for smaller batch sizes.

| Performance (in sec) | VectorWolf | TensorFlow |
|----------|----------|----------|
| [Cancer](https://github.com/soham-c04/VectorWolf/blob/main/Images/CancerPrediction/3.%20Training%20end%20by%20EarlyStopping.png) [Prediction](https://github.com/soham-c04/VectorWolf/blob/main/Kaggle/CancerPrediction/cancer-prediction.ipynb) | 2.226 | 10.115 |
| [Coffee](https://github.com/soham-c04/VectorWolf/blob/main/Images/CoffeeRoasting/3.%20Training%20the%20model.png) [Roasting](https://github.com/soham-c04/VectorWolf/blob/main/Kaggle/CoffeeRoasting/coffee-roasting-tf.ipynb) | 1.238 | 47.201 |
| [House Price](https://github.com/soham-c04/VectorWolf/blob/main/Images/HousePricePrediction/5.%20Train_time%20and%20predicting.png) [Prediction](https://github.com/soham-c04/VectorWolf/blob/main/Kaggle/HousePricePrediction/house-price-prediction.ipynb) | 5.259 | 85.520 |

## Table of Contents
- [**How to use**](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#how-to-use)
- [**Important Points**](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#important-points)
- [**Metric**](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#metric)
  - Classification
    - [Accuracy](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#1-accuracy)
    - [Recall](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#2-recall)
    - [Precision](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#3-precision)
    - [F1 score](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#4-f1_score)
    - [Classification Metrics](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#5-classification_metrics)
    - [Confusion Matrix](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#6-confusion_matrix)
  - Regression
    - [Mean Absolute Error](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#7-mean_absolute_error)
    - [Mean Squared Error](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#8-mean_squared_error)
    - [Root Mean Squared Error](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#9-root_mean_squared_error)
- [**Activation functions**](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#activation-functions)
  - [Linear](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#linear)
  - [ReLu](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#relu)
  - [Sigmoid](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#sigmoid)
- [**Loss functions**](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#losses)
  - [Mean Squared Error](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#meansquarederror)
  - [Binary Cross Entropy](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#binarycrossentropy)
- [**Optimizers**](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#optimizers)
  - [SGD](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#sgd) (Stochastic Gradient Descent)
  - [Adam](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#adam) (Adaptive Moment Estimation)
- [**Callbacks**](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#callbacks)
  - [EarlyStopping](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#earlystopping)
- [**History**](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#history)
- [**Methods for Layer**](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#methods-for-layer--)
  - [Dense](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#1-layersdense)
  - [operator](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#2-operator)
  - [info](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#3-layerinfo)
  - [get_name](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#4-layerget_name)
  - [set_name](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#5-layerset_name)
  - [get_units](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#6-layerget_units)
  - [get_weights](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#7-layerget_weights)
  - [set_weights](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#8-layerset_weights)
  - [get_bias](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#9-layerget_bias)
  - [set_bias](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#10-layerset_bias)
  - [activation_name](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#11-layeractivation_name)
  - [print(layer)](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#12-printlayer)
- [**Methods for Model**](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#methods-for-model--)
  - [Sequential](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#1-modelssequential)
  - [add](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#2-modeladd)
  - [summary](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#3-modelsummary)
  - [compile](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#4-modelcompile)
  - [fit](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#5-modelfit)
  - [predict](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#6-modelpredict)
  - [evaluate](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#7-modelevaluate)
  - [set_features](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#8-modelset_features)
  - [get_layer](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#9-modelget_layer)
  - [get_layers](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#10-modelget_layers)
  - [print(model)](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#11-printmodel)
- [**Other Methods**](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#other-methods)
  - [Read CSV](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#1-read_csv)
  - [Write CSV](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#2-write_csv)
  - [Shape](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#3-shape)
  - [Tranpose](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#4-transpose)
  - [Multiply](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#5-multiply)
  - [Hadamard Product](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#6-hadamard_product)
  - [Print](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#7-print)

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

## Metric

**Parameters:-**
- `y_true`: Actual results (necessary).
- `y_pred`: Predicted results (necessary).
- `print_`: Whether to print the result (set to **True** by default) (optional).
<br>

**Use**: Gets the evaluation metric values after model prediction.

**Add `metrics.` before calling any of the Metric methods.**

### 1. accuracy
> $\text{Accuracy} = \frac{\text{correct classifications}}{\text{total classifications}} = \frac{TP+TN}{TP+TN+FP+FN}$
> 
> **returns** -> `D`

### 2. recall
> $\text{Recall (or TPR)} = \frac{\text{correctly classified actual positives}}{\text{all actual positives}} = \frac{TP}{TP+FN}$
> 
> **returns** -> `D`

### 3. precision
> $\text{Precision} = \frac{\text{correctly classified actual positives}}{\text{everything classified as positive}} = \frac{TP}{TP+FP}$
> 
> **returns** -> `D`

### 4. f1_score
> $\text{F1-score}=2*\frac{\text{precision * recall}}{\text{precision + recall}} = \frac{2\text{TP}}{2\text{TP + FP + FN}}$
> 
> **returns** -> `D`

### 5. classification_metrics
> **returns** -> `void`
> 
> **Use**: Prints all the above metrics.

### 6. confusion_matrix
> **returns** -> `vector<vector<int>>` (2 $\times$ 2)

### 7. mean_absolute_error
> $\text{MAE}(y,\hat{y}) = \frac{1}{m} \cdot \sum\limits_{i=1}^{m}|y_i - \hat{y}_i|$
> 
> **returns** -> `D`

### 8. mean_squared_error
> $\text{MSE}(y,\hat{y}) = \frac{1}{m} \cdot \sum\limits_{i=1}^{m}(y_i - \hat{y}_i)^2$
> 
> **returns** -> `D`

### 9. root_mean_squared_error
> $\text{RMSE}(y,\hat{y}) = \sqrt{ \frac{1}{m} \cdot \sum\limits_{i=1}^{m}(y_i - \hat{y}_i)^2 }$
> 
> **returns** -> `D`
<br>

## Activation functions

Parameter in [layers.Dense()](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#1-layersdense).<br>
Pass it as `activation = "activation_name"` (case independent).

### Linear
> $f(z) = z$
  
### ReLu
> $f(z) = z \geq 0$

### Sigmoid
> $f(z) = \frac{1}{1+e^{(-z)}}$
<br>

## Losses

Parameter in [model.compile()](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#4-modelcompile).<br>
Pass it as `loss = "loss_name"` (case independent).

### MeanSquaredError
> $ℒ(y, \hat{y}) = \frac{1}{m} \cdot \sum\limits_{i=1}^{m} (y_i - \hat{y}_i)^2$
>
> Can also be passed as "mse"
  
### BinaryCrossentropy
> $ℒ(y, \hat{y}) = -\frac{1}{m} \cdot \sum\limits_{i=1}^{m} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$
>
> Can also be passed as "bce"
<br>

## Optimizers

Parameter in [model.compile()](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#4-modelcompile).<br>
Pass it as `optimizer = "optimizer_name"` (case independent).

### SGD
> This is the default optimizer if not mentioned.
> **Hyperparameters:-**
> - `learning_rate`: Learning rate (set to **0.001** by default) (optional)
  
### Adam
> **Hyperparameters:-**
> - `learning_rate`: Learning rate (set to **0.001** by default) (optional).
> - `beta_1`: Exponents decay rate for first moment (set to **0.9** by default) (optional).
> - `beta_2`: Exponents decay rate for second moment (variance) (set to **0.999** by default) (optional).
> - `epsilon`: For numerical stability (prevent division by zero) (set to $10^{-7}$ by default) (optional).
<br>

## Callbacks

Parameter in [model.fit()](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#5-modelfit).<br>
Pass it as `callbacks = {callback1, callback2, ...}`.

**Add `Callback::` before the Callback methods.**

### EarlyStopping
> **returns** -> `class Callback`
>
> **Use:** Stops training with when given condition is true.
> 
> **Parameters:-**
> - `monitor`: Parameter on which condition is set (necessary).
> - `mode`: Given parameter to be **"min"** or **"max"** (If not mentioned automatically set as per parameter) (optional).
> - `patience`: No. of epochs after which training stops, if monitored parameter is not optimized (necessary).
<br>

## History

Struct returned by [model.fit()](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#5-modelfit).<br>
Contains information about training the model.

**Attributes:-**
- `epoch`: `vector<int>` of epochs on which model was trained.
- `history`: `map<string,vector<D>>`. Contains the history of losses and val_losses (if present).
- `params`: `map<string,int>`. Contains number of epochs and steps_per_epoch of model trained.
<br>

## Methods for Layer -

### 1. layers.Dense()
> **returns** -> `class Layer`
>  
> **Use:** Creating a layer.
> 
> **Parameters:-**
> - `units`: No. of units in that layer (necessary).
> - [Activation](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#activation-functions) (set to **linear** by default) (optional).
> - `name`: Name of layer (set to **Layer 'layer_number'** by default) (optional).

### 2. `operator()`
> **returns** -> `vector<vector<D>>`
> 
> **Use**: Getting the output Matrix from layer, after giving input. 
> 
> **Parameters**:-
> - `x`: Input for the layer (necessary).
> - `z_store`: If **true**, stores the Matrix z, i.e. just before applying activation function on the Matrix. (set to **false** by default) (optional).

### 3. layer.info()
> **returns** -> `int`
> 
> **Use**: Prints layer name, type, units, parameters for that layer. Returns the parameters in that layer.
> 
> **Parameters**:-
> - `prev_units`: No. of units in previous layer (necessary).

### 4. layer.get_name()
> **returns** -> `string`
> 
> **Use**: Getting the name of layer. 
> 
> **Parameters**: None

### 5. layer.set_name()
> **returns** -> `void`
> 
> **Use**: Setting/Changing the name of layer. 
> 
> **Parameters**:-
> - `name__`: Updated name of the layer (necessary).

### 6. layer.get_units()
> **returns** -> `int`
> 
> **Use**: Getting the number of units in the layer. 
> 
> **Parameters**: None

### 7. layer.get_weights()
> **returns** -> `vector<vector<D>>`
> 
> **Use**: Getting the weight Matrix of layer. 
> 
> **Parameters**: None

### 8. layer.set_weights()
> **returns** -> `void`
> 
> **Use**: Load a weight matrix onto the layer.
> 
> **Parameters**:-
> - `new_weight`: Updated weight matrix of the layer (necessary).

### 9. layer.get_bias()
> **returns** -> `vector<D>`
> 
> **Use**: Getting the bias vector of layer. 
> 
> **Parameters**: None

### 10. layer.set_bias()
> **returns** -> `void`
> 
> **Use**: Load a bias vector onto the layer. 
> 
> **Parameters**:-
> - `new_bias`: Updated bias vector of the layer (necessary).

### 11. layer.activation_name
> **returns** -> `string &`
> 
> **Use**: Getting the name of activation function of the layer.
> 
> **Parameters**: None
  
### 12. print(layer)
> **returns** -> `void`
> 
> **Use**: Prints layer_name, units, all weights and biases for that layer.
> 
> **Parameters**:-
> - `layer`: class Layer to be printed (necessary).
<br>

## Methods for Model -

### 1. models.Sequential();
> **returns** -> `class Model`
> 
> **Use**: Creating a model.
> 
> **Parameters**:-
> - `input_param`: No. of features in input of training data (necessary).
> - `vector<Layer>`: Vector of layers inputed in the form ```{ layers.Dense(units = ..., activation = "...", name = "..."), ... }``` (optional).

### 2. model.add();
> **returns** -> `void`
> 
> **Use**: Adding a new layer to the model.
> 
> **Parameters**:-
> - `new_layer`: New layer to be added to be model (necessary).

### 3. model.summary()
> **returns** -> `void`.
> 
> **Use**: Prints the layer_name, output shape and parameters for each layer in model
> 
> **Parameters**: None

### 4. model.compile()
> **returns** -> `void`.
> 
> **Use**: Setting the loss function and learning_rate for the model.
> 
> **Parameters**:-
> - [Loss](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#losses) (set to **MeanSquaredError** by default) (optional).
> - [Optimizer](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#optimizers) (set to **SGD** by default) (optional).

### 5. model.fit()
> **returns** -> `struct History`.
> 
> **Use**: Training the model. Prints the current epoch_number, Loss (and val_loss if available) for that epoch and Time to run that epoch.
> 
> **Parameters**:-
> - `x_train`: Input for training data (necessary).
> - `y_train`: Ouput for given data (necessary).
> - `epochs`: No. of epochs (set to **0** i.e. nothing happens by default) (optional).
> - `batch_size`: Input data divided into subsets (set to **32** by default) (If **$\text{batch\\_size} \nmid \text{|x\\_train|}$**, it warps around from start to compensate) (optional).
> - `steps_per_epoch`: No. of steps per epoch (set to **$\left\lceil \frac{|{x\\_train}|}{batch\\_size} \right\rceil$** by default) (optional).
> - `validation_data`: Pair of {X_test,y_test} to compare test losses (optional)
> - [Callbacks](https://github.com/soham-c04/VectorWolf?tab=readme-ov-file#callbacks)
> - `Shuffle`: Training data will be randomly shuffled before each epoch (set to **true** by default) (optional).

### 6. model.predict();
> **returns** -> `vector<D>`
> 
> **Use**: Predicting the output from input Matrix based on the pre-compiled and fitted model. Returns output vector.
> 
> **Parameters**:-
> - `x`: Input to be predicted/processed (necessary).
> - `print`: Whether the to print box for predict (set to **true** by default) (optional).

### 7. model.evaluate();
> **returns** -> `D`
> 
> **Use**: Calculating the loss on a given dataset based on previous training of model.
> 
> **Parameters**:-
> - `X_test`: Input for the dataset (necessary).
> - `y_test`: Output for the dataset (necessary).
> - `print_`: Whether the to print box for predict (set to **true** by default) (optional).

### 8. model.set_features();
> **returns** -> `void`
> 
> **Use**: Updating the no. of input features for the model.
> 
> **Parameters**:-
> - `input_features_`: New no. of input features for the model.

### 9. model.get_layer();
> **returns** -> `class Layer &`
> 
> **Use**: Getting a layer by reference to do changes to it or get information.
> 
> **Parameters**:-
> - `name_`: Name of layer to be returned (necessary).

### 10. model.get_layers();
> **returns** -> `vector<class Layer>`
> 
> **Use**: Getting all the layers in a model.
> 
> **Parameters**: None
  
### 11. print(model)
> **returns** -> `void`.
> 
> **Use**: Prints [print(layer[i])](https://github.com/soham-c04/VectorWolf/tree/main?tab=readme-ov-file#printlayer) for all layers.
> 
> **Parameters**:-
> - `model`: class Model to be printed (necessary).
<br>

## Other Methods

### 1. read_csv();
> **returns** -> `vector<vector<D>>`
> 
> **Use**: Reads input/ouput data from a .csv file and returns as a 2D vector.
> 
> **Parameters**:-
> - `path`: Absolute/relative path to the .csv file to be read (necessary).
> - `header`: To detect if headers (1st row) is to be filtered out separately (set to **true** by default) (optional).
> - `dummy_replace`: `map<string,map<string,D>>`. For replacing dummy variables from columns from string to D (optional).
> - `null_values`: vector of strings which are treated as NULL values (set to **empty string** by default) (optional).

### 2. write_csv();
> **returns** -> `void`
> 
> **Use**: Writing data onto a .csv file for further processing later.
> 
> **Parameters**:-
> - `path`: Absolute/relative path to the .csv file to be read (necessary).
> - `data`: vector to be written into .csv file (necessary).

### 3. shape();
> **returns** -> `void`
> 
> **Use**: Printing the shape of a Matrix.
> 
> **Parameters**:-
> - `M`: Matrix whose shape is to be printed (necessary).

### 4. transpose();
> **returns** -> `vector<vector<D>>`
> 
> **Use**: Getting the transpose of a Matrix.
> 
> **Parameters**:-
> - `M`: Matrix to be transposed (necessary).

### 5. multiply();
> **returns** -> `vector<vector<D>>` or `vector<D>`
> 
> **Use**: Multiplying two Matrices (if their dimensions match) **or** Matrix with a constant **or** Vector with a constant.
> 
> **Parameters**:-
> - `a`: First Matrix or Vector (necessary).
> - `b` or `c`: Second Matrix or constant (necessary).

### 6. hadamard_product();
> **returns** -> `vector<vector<D>>` or `vector<D>`
> 
> **Use**: Element-wise product of two Matrices  **or** Vectors (if their dimensions match).
> 
> **Parameters**:-
> - `a`: First Matrix or Vector (necessary).
> - `b`: Second Matrix or Vector (necessary).

### 7. print();
> **returns** -> `void`
> 
> **Use**: Printing contents of a 1D vector, 2D vector, Layer or Model.
> 
> **Parameters**:-
> - `vec` or `mat` or `layer` or `model`: Any one of the four (necessary).



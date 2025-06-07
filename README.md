# VectorWolf
**VectorWolf is (Tensor => Vector) + (reverse(Flow) => Wolf)**

Syntax is almost the same as TensorFlow.

## Methods for Layer -
### layers.Dense()
**returns** -> `class Layer`

Parameters:-
-  `units`: No. of units in that layer (necessary).
-  `activation`: Activation function for that layer (set to **linear** by default) (optional).
-  `name`: Name of layer (set to **Layer 'layer_number'** by default) (optional).

### print(layer)
**returns** -> `void`.

Prints layer_name, units, all weights and biases for that layer.

Parameters:-
- `layer`: class Layer to be printed (necessary).

## Methods for Model -
### keras.Sequential();
**returns** -> `class Model`

Parameters:-
-  `input_param`: No. of features in input of training data (necessary).
-  `vector<Layer>`: Vector of layers inputted in the form ```{ layers.Dense(units = ..., activation = "...", name = "..."), ... }``` (necessary).
### model.summary()
**returns** -> `void`.

Prints the layer_name, output shape and parameters for each layer in model

Parameters: None

### model.compile()
**returns** -> `void`.

Parameters:-
- `loss`: Loss function used in model (set to **MeanSquaredError** by default) (optional).
- `learning_rate`: learning_rate for weights/bias (set to **0** i.e. no updates by default) (optional).

### model.fit()
**returns** -> `void`.

Prints the current_epoch number, Loss for that epoch and Time for that epoch.

Parameters:-
- `x_train`: Input for training data (necessary).
- `y_train`: Ouput for given data (necessary).
- `epochs`: No. of epochs (set to **0** i.e. nothing happens by default) (optional).
- `batch_size`: Input data divided into subsets (set to **32** by default) (If $\text{batch_ size} \nmid |\text{x_train}|$, it warps around from start to compensate) (optional).
- `steps_per_epoch`: No. of steps per epoch (set to $\left\lceil \frac{|{\text{x_train}}|}{\text{batch_size}} \right\rceil$ by default) (optional).
- `Shuffle`: Training data will be randomly shuffled before each epoch (set to **true** by default) (optional).

### print(model)
**returns** -> `void`.

Prints - [print(layer[i])](https://github.com/soham-c04/VectorWolf/tree/main?tab=readme-ov-file#printlayer) for all layers.

Parameters:-
- `model`: class Model to be printed (necessary).

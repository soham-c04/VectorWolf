#include "VectorWolf.h"
#include "Headers/basic.cpp"
// class Metrics

// Error metrics for classification
D Metric::accuracy(vector<D> &y_true, vector<D> &y_pred, bool print_){
	if(print_){
		cout<<endl;
		print_header("Accuracy");
	
		print_top();
	}
	
	int m = y_true.size();
	D accuracy = 0;
	for(int i=0;i<m;i++)
		if((y_pred[i] >= 0.5) == (y_true[i] >= 0.5))
			accuracy += 100;

	accuracy /= m;
	
	if(print_){
		string line = to_string(accuracy) + " %";
		print(line);

		print_bottom();
		cout<<endl;
	}
	
	return accuracy;
}

D Metric::recall(vector<D> &y_true, vector<D> &y_pred, bool print_){
	if(print_){
		cout<<endl;
		print_header("Recall");

		print_top();
	}

	int m = y_true.size(), total_true_positive = 0;
	D recall = 0;
	for(int i=0;i<m;i++){
		if(y_true[i] >= 0.5){
			total_true_positive++;
			if(y_pred[i] >= 0.5)
				recall += 100;
		}
	}

	recall /= total_true_positive;

	if(print_){
		string line = to_string(recall) + " %";
		print(line);

		print_bottom();
		cout<<endl;
	}
	
	return recall;
}

D Metric::precision(vector<D> &y_true, vector<D> &y_pred, bool print_){
	if(print_){
		cout<<endl;
		print_header("Precision");

		print_top();
	}

	int m = y_true.size(), total_pred_positive = 0;
	D precision = 0;
	for(int i=0;i<m;i++){
		if(y_pred[i] >= 0.5){
			total_pred_positive++;
			if(y_true[i] >= 0.5)
				precision += 100;
		}
	}

	precision /= total_pred_positive;

	if(print_){
		string line = to_string(precision) + " %";
		print(line);

		print_bottom();
		cout<<endl;
	}
	
	return precision;
}

D Metric::f1_score(vector<D> &y_true, vector<D> &y_pred, bool print_){
	if(print_){
		cout<<endl;
		print_header("F1-score");

		print_top();
	}

	int m = y_true.size(), denominator = 0;
	D f1_score = 0; // f1_score = 2/(1/recall(y_true, y_pred, false) + 1/precision(y_true, y_pred, false));
	for(int i=0;i<m;i++){
		if(y_true[i] >= 0.5){
			denominator++;
			if(y_pred[i] >= 0.5){
				denominator++;
				f1_score += 200;
			}
		}
		else if(y_pred[i] >= 0.5)
			denominator++;
	}

	f1_score /= denominator;

	if(print_){
		string line = to_string(f1_score);
		print(line);

		print_bottom();
		cout<<endl;
	}

	return f1_score;
}

void Metric::classification_metrics(vector<D> &y_true, vector<D> &y_pred){
	cout<<endl;
	print_header("Classification Metrics");

	print_top();

	int m = y_true.size(), total_true_positive = 0, total_pred_positive = 0;
	D accuracy = 0, true_positive = 0;
	for(int i=0;i<m;i++){
		if(y_true[i] >= 0.5){
			total_true_positive++;
			if(y_pred[i] >= 0.5){
				total_pred_positive++;
				accuracy += 100;
				true_positive += 100;
			}
		}
		else{
			if(y_pred[i] >= 0.5)
				total_pred_positive++;
			else
				accuracy += 100;
		}
	}

	accuracy /= m;
	
	D recall = true_positive/total_true_positive, precision = true_positive/total_pred_positive;
	D f1_score = 2*true_positive/(total_pred_positive + total_true_positive);
	
	string line;
	print(line = "Accuracy  = " + to_string(accuracy) + " %");
	print(line = "");
	print(line = "Recall    = " + to_string(recall) + " %");
	print(line = "");
	print(line = "Precision = " + to_string(precision) + " %");
	print(line = "");
	print(line = "F1-score  = " + to_string(f1_score));

	print_bottom();
	cout<<endl;
}

// Error metrics for regression
D Metric::mean_absolute_error(vector<D> &y_true, vector<D> &y_pred, bool print_){
	if(print_){
		cout<<endl;
		print_header("Mean Absolute Error");

		print_top();
	}

	int m = y_true.size();
	D mae = 0;
	for(int i=0;i<m;i++)
		mae += abs(y_true[i] - y_pred[i]);
	mae /= m;

	if(print_){
		string line = to_string(mae);
		print(line);

		print_bottom();
		cout<<endl;
	}

	return mae;
}

D Metric::mean_squared_error(vector<D> &y_true, vector<D> &y_pred, bool print_){
	if(print_){
		cout<<endl;
		print_header("Mean Squared Error");

		print_top();
	}

	int m = y_true.size();
	D mse = 0;
	for(int i=0;i<m;i++)
		mse += (y_true[i] - y_pred[i])*(y_true[i] - y_pred[i]);
	mse /= m;

	if(print_){
		string line = to_string(mse);
		print(line);

		print_bottom();
		cout<<endl;
	}

	return mse;
}

D Metric::root_mean_squared_error(vector<D> &y_true, vector<D> &y_pred, bool print_){
	if(print_){
		cout<<endl;
		print_header("Root Mean Squared Error");

		print_top();
	}

	D rmse = sqrtl(mean_squared_error(y_true, y_pred, false));

	if(print_){
		string line = to_string(rmse);
		print(line);

		print_bottom();
		cout<<endl;
	}

	return rmse;
}

Metric metrics;

// class Activation

D Activation::linear(const D &t){
	return t;
}

D Activation::ReLu(const D &t){
	if(t<0) return 0;
	return t;
}

D Activation::sigmoid(const D &t){
	return 1/(1+expl(-t));
}

D Activation::deriv_linear(const D &t){
	return 1;
}

D Activation::deriv_ReLu(const D &t){
	if(t<=0) return 0;
	return 1;
}

D Activation::deriv_sigmoid(const D &t){
	D sig=sigmoid(t);
	return sig*(1-sig);
}

// class Loss

D Loss::MeanSquaredError(vector<D> &y, vector<D> &a){
	int m = y.size();
	D loss=0;
	for(int i=0;i<m;i++) loss+=(a[i]-y[i])*(a[i]-y[i]);
	loss /= m;
	return loss;
}

D Loss::BinaryCrossentropy(vector<D> &y, vector<D> &a){
	int m = y.size();
	D loss=0;
	for(int i=0;i<m;i++){
		D y1 = a[i];
		if(y1<eps) y1=eps;
		else if(y1>1-eps) y1=1-eps;

		if(y[i]) loss += -(D)logl(y1);
		else loss += -(D)logl(1-y1);
	}
	loss /= m;
	return loss;
}

vector<vector<D>> glorot_uniform(int n,int m){ // Shape = (a,b)
	D limit = sqrtl((D)6/(n+m));

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<D> rand_num(-limit, limit);

    vector<vector<D>> weight(n,vector<D>(m));

    for(int i=0;i<n;i++)
		for(int j=0;j<m;j++)
			weight[i][j]=rand_num(gen);

    return weight;
}

/*
Syntax:-
layers.Dense(
    units,                      # Number of neurons (REQUIRED)
    activation=None,            # Activation function
    use_bias=True,              # Whether to use a bias term
    kernel_initializer="glorot_uniform",  # How to initialize weights
    bias_initializer="zeros",   # How to initialize bias
    kernel_regularizer=None,    # Regularization (L1, L2, etc.)
    bias_regularizer=None,      # Regularization for bias
    activity_regularizer=None,  # Regularization for output
    kernel_constraint=None,     # Constraint on weights (e.g., max norm)
    bias_constraint=None,       # Constraint on bias
    name=None                   # Name of the layer
)
*/

// Keyword arguments for layers.Dense()

int units = 0;
string activation = "";
string name = "";

// class Layer

Layer::Layer():units_(0){};

Layer::Layer(const int units__,const string activation__,const string name__,const string type_):units_(units__),name_(name__),type(type_),
																						  activation_name(lower_case(activation__)){
	if(units_<=0){
		cout<<"\nWarning: Layer '"<<name_<<"' dicarded.\n";
		cout<<"Reason: <=0 units"<<endl;
		return;
	}

	// Set Layer Name
	static int layer_number = 0; // Used for nomenclature of layers (default name)
	layer_number++;

	if(name_==""){
		name_ = "Layer"+to_string(layer_number);
		cout<<"\nLayer name not given.\n";
		cout<<"Default name set to - "<<name_<<endl;
	}
	while(name_.size()<7) name_.push_back(' ');

	// Set Activation
	if(activation_name == "linear"){
		activation_ = linear;
		deriv_act	= deriv_linear;
	}
	else if(activation_name == "relu"){
		activation_ = ReLu;
		deriv_act	= deriv_ReLu;
	}
	else if(activation_name == "sigmoid"){
		activation_ = sigmoid;
		deriv_act	= deriv_sigmoid;
	}
	else{
		cout<<"\nWarning: Input activation function is not present.\n";
		cout<<"Default activation function - 'linear' will be used."<<endl;
		activation_name = "linear";
		activation_ = linear;
		deriv_act	= deriv_linear;
	}
}

Layer Layer::Dense(int units__, string activation__, string name__){
	if(units>0) units__=units;
	if(activation != "") activation__=activation;
	if(name != "") name__=name;

	// Reset global variables
	units	   = 0;
	name  	   = "";
	activation = "";
	return Layer(units__,activation__,name__,"Dense");
}

vector<vector<D>> Layer::operator()(vector<vector<D>> &x, bool z_store){
	// Randomly initialize weights if layer not yet
	int n=x.size(),m=x[0].size();
	if(weight.empty()){
		weight=glorot_uniform(units_,n);
		bias.resize(units_);
	}

	vector<vector<D>> output = multiply(weight,x);
	if(z_store) z = output;
	for(int i=0;i<units_;i++){
		for(int j=0;j<m;j++){
			output[i][j] += bias[i];
			if(z_store) z[i][j] = output[i][j];
			output[i][j] = activation_(output[i][j]);
		}
	}
	return output;
}

void Layer::element_wise_product(vector<vector<D>> &dJ_dz){
	int n = dJ_dz.size(), m = dJ_dz[0].size();
	for(int i=0;i<n;i++)
		for(int j=0;j<m;j++)
			dJ_dz[i][j] *= deriv_act(z[i][j]);
}

void Layer::update_weights(vector<vector<D>> &dJ_dw){
	int p = weight.size(),q = weight[0].size();
	for(int i=0;i<p;i++)
		for(int j=0;j<q;j++)
			weight[i][j] -= dJ_dw[i][j];
}

void Layer::update_bias(vector<D> &dJ_db){
	int n = weight.size();
	for(int i=0;i<n;i++)
		bias[i] -= dJ_db[i];
}

int Layer::info(int prev_units){
	string line = name_ + '(' + type + ")       (," + to_string(units_) + ")          " + to_string(units_*(1+prev_units));
	print(line);
	return units_*(1+prev_units);
}

string Layer::get_name(){
	return name_;
}

void Layer::set_name(const string name__){
	name_ = name__;
}

int Layer::get_units(){
	return units_;
}

vector<vector<D>> Layer::get_weights(){
	return weight;
}

void Layer::set_weights(vector<vector<D>> new_weight){
	swap(weight,new_weight);
}

vector<D> Layer::get_bias(){
	return bias;
}

void Layer::set_bias(vector<D> new_bias){
	swap(bias,new_bias);
}

Layer layers;

// class Model

Model::Model():layers(),Learning_rate(0),input_features(0){};

void Model::add(Layer new_layer){
	if(new_layer.get_units()>0) layers.push_back(new_layer);
	else{
		string line = "Layer should have at least 1 unit.";
		print(line);
	}
}

// Keyword arguments for model.Sequential()

int input_param = 0;

Model Model::Sequential(int input_features_, vector<Layer> layers_){
	cout<<endl;

	string line;

	// Box for keras.summary()
	print_header("models.Sequential()");

	// Box for output of keras.Sequential()
	print_top();

	Model my_model;
	if(input_param > 0) input_features_ = input_param;
	if(input_features_ > 0) my_model.set_features(input_features_);
	for(Layer &my_layer:layers_) my_model.add(my_layer);
	input_param = 0;

	print_bottom();
	cout<<endl;
	return my_model;
}

void Model::summary(){
	cout<<endl;

	string line;

	// Box for model.summary()
	print_header("model.summary()");

	// Box for output of model.summary()
	print_top();


	for(int i=0;i<print_width;i++) line.push_back('-');
	print(line);

	line = "Layer   (type)     Output Shape    Param #";
	print(line);

	for(int i=0;i<print_width;i++) line.push_back('=');
	print(line);

	int total_params=0;
	total_params+=layers[0].info(input_features);
	for(int i=1;i<layers.size();i++) total_params+=layers[i].info(layers[i-1].get_units());
	for(int i=0;i<print_width;i++) line.push_back('=');
	print(line);

	line = "Total Parameters: " + to_string(total_params);
	print(line);

	for(int i=0;i<print_width;i++) line.push_back('-');
	print(line);

	print_bottom();
	cout<<endl;
}

// Keyword arguments for model.compile()

string loss = "";
double learning_rate = 0;

// Keyword arguments for model.fit()

int epochs = 0;
int batch_size = 32;
int steps_per_epoch = 0;
bool Shuffle = true;
pair<vector<vector<D>>,vector<D>> validation_data;

void reset_fit(){   // Resets global variable values
	epochs = 0;
	batch_size = 32;
	steps_per_epoch = 0;
	Shuffle = true;
	validation_data = {{},{}};
}

vector<D> Model::predict(vector<vector<D>> x, bool print_){
	if(print_){
		cout<<endl;
		print_header("model.predict");

		print_top();
	}

	vector<vector<D>> output = x;
	if(x[0].size()==input_features){
		output = transpose(output);
		for(Layer &cur_layer:layers) output = cur_layer(output);
	}
	else{
		string line = "No. of input features should be - " + to_string(input_features) + '.';
		print(line);
	}

	if(print_){
		print_bottom();
		cout<<endl;
	}
	
	return output[0];
}

D Model::evaluate(vector<vector<D>> X_test, vector<D> y_test, bool print_){
	if(print_){
		cout<<endl;
		print_header("model.evaluate");

		print_top();
	}

	vector<D> prediction = predict(X_test, false);
	D eval = loss_func(y_test, prediction);
	
	if(print_){
		string line = to_string(eval);
		print(line);
		
		print_bottom();
		cout<<endl;
	}
	
	return eval;
}

void Model::set_features(int input_features_){
	input_features = input_features_;
}

Layer& Model::get_layer(string name_){
	while(name_.size()<7) name_.push_back(' ');

	for(Layer &l:layers)
		if(l.get_name() == name_) return l;

	cout<<endl;

	// Box for model.get_layer()
	print_header("model.get_layer()");

	// Box for output of model.get_layer()
	print_top();

	string line = "Layer not found.";
	print(line);

	print_bottom();
	cout<<endl;
}

vector<Layer> Model::get_layers(){
	return layers;
}

Model models;

void print(Layer &l){
	// Box for Header as layer name
	string header = "";
	header = header + VERT + "  Layer name: " + l.get_name() + "  " + VERT;
	int w = header.size();

	string line = "";
	line = line + TOP_LEFT;
	for(int p=0;p<w-2;p++) line.push_back(HORIZ);
	line += TOP_RIGHT;
	print(line);

	print(header);

	line = "";
	line = line + BOTTOM_LEFT;
	for(int p=0;p<w-2;p++) line.push_back(HORIZ);
	line += BOTTOM_RIGHT;
	print(line);

	// Box for details of layer
	line = "Units: " + to_string(l.get_units());
	print(line);

	line = "Weights: ";
	print(line);
	print(l.get_weights());

	line = "Bias: ";
	print(line);
	print(l.get_bias());
	print(line = ""); print(line = "");
}

void print(Model &m){
	cout<<endl;

	// Box for print(model)
	print_header("print(model)");

	// Box for output of print(model)
	print_top();

	vector<Layer> layers = m.get_layers();
	for(Layer &l:layers) print(l);

	print_bottom();
	cout<<endl;
}

/*
Use PCA to filter out unnecessary features to make computation faster ?
Normalize mean + variance ?
*/

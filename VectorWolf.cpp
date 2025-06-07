#include "VectorWolf.h"
#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <sstream>
#include <functional>
#include <random>
using namespace std;

// long double -- precision of 15-33 decimal places -- But Slower -- 8 to 16 bytes
// double -- precision upto 15 decimal places -- Medium -- 8 bytes
// float -- precision upto 6 decimal places -- Relatively fast -- 4 bytes
using D = double; // Change it to: using D = data_type;

string lower_case(const string &s){
	string low="";
	for(char c:s) low.push_back(c|32);
	return low;
}

void print(const vector<D> &vec){
	for(D a:vec) cout<<a<<" ";
	cout<<endl;
}

void print(const vector<vector<D>> &vec){
	for(vector<D> v:vec) print(v);
}

void shape(vector<vector<D>> &M){
	cout<<"\n(";
	if(M.empty()) cout<<",";
	else{
		cout<<M.size()<<",";
		if(M[0].size() != 0) cout<<M[0].size();
	}
	cout<<")"<<endl;
}

vector<vector<D>> transpose(vector<vector<D>> &M){
	int n=M.size(),m=M[0].size();
	vector<vector<D>> M_t(m,vector<D>(n));
	for(int i=0;i<n;i++)
		for(int j=0;j<m;j++)
			M_t[j][i]=M[i][j];
			
	return M_t;
}

vector<vector<D>> multiply(vector<vector<D>> &a, vector<vector<D>> &b){
	int p = a.size(), q = a[0].size(), r = b[0].size();
	vector<vector<D>> ans(p,vector<D>(r));
	
	if(q != b.size()) cout<<"\nDimension mismatch.\n";
	else{
		for(int i=0;i<p;i++){
			for(int j=0;j<r;j++){
				D ans_ij = 0;
				for(int k=0;k<q;k++) ans_ij += a[i][k]*b[k][j];
				ans[i][j] = ans_ij;
			}
		}
	}
	return ans;
}

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

D Loss::MeanSquaredError(vector<D> &y,vector<D> &a){
	int m = y.size();
	D loss=0;
	for(int i=0;i<m;i++) loss+=(a[i]-y[i])*(a[i]-y[i]);
	loss /= 2*m;
	return loss;
}

D Loss::BinaryCrossentropy(vector<D> &y,vector<D> &a){
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

vector<vector<D>> glorot_uniform(int n,int m){ // Shape = (a,b)
	D limit = sqrtl(6/(n+m));

    random_device rd;
    mt19937 gen(rd()); 
    uniform_real_distribution<D> rand_num(-limit, limit);

    vector<vector<D>> weight(n,vector<D>(m));

    for(int i=0;i<n;i++) for(int j=0;j<m;j++) weight[i][j]=rand_num(gen);

    return weight;
}

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

vector<vector<D>>& Layer::operator()(vector<vector<D>> &x, bool z_store){
	// Randomly initialize weights if layer not yet 
	int n=x.size(),m=x[0].size();
	if(weight.empty()){
		weight=glorot_uniform(units_,n);
		bias.resize(units_);
	}

	output = multiply(weight,x);
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
	int m = 0;
	if(output.size()) m = output[0].size();
	cout<<name_<<" ("<<type<<")       ("<<m<<","<<units_<<")          "<<units_*(1+prev_units)<<endl;
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

void Layer::set_units(int units__){
	units_ = units__;
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

vector<vector<D>> Layer::get_output(){
	return output;
}

Layer layers;

// class Model

Model::Model():layers(),Learning_rate(0),input_features(0){};
		
void Model::add(Layer new_layer){
	if(new_layer.get_units()>0) layers.push_back(new_layer);
	else cout<<"\nLayer should have at least 1 unit.\n";
}

// Keyword for model.Sequential()

int input_param = 0;

Model Model::Sequential(int input_features_, vector<Layer> layers_){
	Model my_model;
	if(input_param > 0) input_features_ = input_param;
	input_param = 0;
	if(input_features_ > 0) my_model.set_features(input_features_);
	for(Layer &my_layer:layers_) my_model.add(my_layer);
	return my_model;
}

void Model::summary(){
	cout<<endl;
	for(int i=0;i<print_width;i++) cout<<"-"; cout<<endl;
	cout<<"Layer   (type)     Output Shape    Param #"<<endl;
	for(int i=0;i<print_width;i++) cout<<"="; cout<<endl;

	int total_params=0;
	total_params+=layers[0].info(input_features);
	for(int i=1;i<layers.size();i++) total_params+=layers[i].info(layers[i-1].get_units());
	for(int i=0;i<print_width;i++) cout<<"="; cout<<endl;
	
	cout<<"Total Parameters: "<<total_params<<endl;
	for(int i=0;i<print_width;i++) cout<<"-"; cout<<endl;
}

// Keyword arguments for model.compile()

string loss = "";
double learning_rate = 0;

// Keyword arguments for model.fit()

int epochs = 0;
int batch_size = 0;
int steps_per_epoch = 0;
bool Shuffle = true;

void reset_fit(){   // Resets global variable values
	epochs = 0;
	batch_size = 0;
	steps_per_epoch = 0;
	Shuffle = true;
}

vector<D> Model::predict(const vector<vector<D>> x){
	vector<vector<D>> output = x;
	if(x[0].size()==input_features){
		output = transpose(output);
		for(Layer &cur_layer:layers) output = cur_layer(output);
	}
	else cout<<"\nNo. of input features should be - "<<input_features<<"."<<endl;
	return output[0];
}

void Model::set_features(int input_features_){
	input_features = input_features_;
}

Layer& Model::get_layer(string name_){
	while(name_.size()<7) name_.push_back(' ');
	
	for(Layer &l:layers)
		if(l.get_name() == name_) return l;
	cout<<"\nLayer not found."<<endl;
}

vector<Layer> Model::get_layers(){
	return layers;
}

Model keras;

void print(Layer &l){
	cout<<"\nLayer name: "<<l.get_name();
	cout<<"\n\nUnits: "<<l.get_units();
	cout<<"\n\nWeights: \n";
	print(l.get_weights());
	cout<<"\nBias: ";
	print(l.get_bias());
}

void print(Model &m){
	vector<Layer> layers = m.get_layers();
	for(Layer &l:layers) print(l);
}

vector<vector<D>> read_csv(const string path){
	ifstream file(path);
    string line;

    vector<vector<D>> data;

    while (getline(file, line)) {
        vector<D> row;
        stringstream ss(line);
        string value;

        while (getline(ss,value,',')) row.push_back(stod(value));

        data.push_back(row);
    }
    
    return data;
}


/*
Use PCA to filter out unnecessary features to make computation faster ?
Normalize mean + variance ?
Initialize weights & biases as close to 1 to prevent vanishing / exploding gradients
w_i = 1/m
or choose https://www.youtube.com/watch?v=zUazLXZZA2U&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=12
Timestamp: 1:02:00
*/

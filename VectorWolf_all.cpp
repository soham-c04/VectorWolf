// All in one header file

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <math.h>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>
#include <random>
using namespace std;

// long double -- precision of 15-33 decimal places -- But Slower -- 8 to 16 bytes
// double -- precision upto 15 decimal places -- Medium -- 8 bytes
// float -- precision upto 6 decimal places -- Relatively fast -- 4 bytes
using D = double;

// For printing boxes
#define VERT         (char)186
#define HORIZ        (char)205
#define TOP_LEFT     (char)201
#define TOP_RIGHT    (char)187
#define BOTTOM_LEFT  (char)200
#define BOTTOM_RIGHT (char)188
#define SCREEN_WIDTH 130

string lower_case(const string &s){
	string low="";
	for(char c:s) low.push_back(c|32);
	return low;
}

void print(string &line, int width = SCREEN_WIDTH){
	int l = line.size();
	cout<<VERT<<"  "<<line;
	for(int p=0;p<width-l-2;p++) cout<<" ";
	cout<<VERT<<"\n";
	line.clear();
}

void print_top(int width = SCREEN_WIDTH){
	cout<<TOP_LEFT;
	for(int c=0;c<width;c++) cout<<HORIZ;
	cout<<TOP_RIGHT<<"\n";
	if(width == SCREEN_WIDTH){
		string line;
		print(line = "");
	}
}

void print_bottom(int width = SCREEN_WIDTH){
	if(width == SCREEN_WIDTH){
		string line;
		print(line = "");
	}
	cout<<BOTTOM_LEFT;
	for(int c=0;c<width;c++) cout<<HORIZ;
	cout<<BOTTOM_RIGHT<<endl;
}

void print_header(string line){
	int l = line.size();
	print_top(l + 4);
	print(line,l + 4);
	print_bottom(l + 4);
}

void print(const vector<D> &vec){
	string line = "      ";
	for(D a:vec) line += to_string(a) + ' ';
	print(line);
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

struct Metric{
	// Error metrics for classification

	D accuracy(vector<D> &y_true, vector<D> &y_pred, bool print_ = true){
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

	D recall(vector<D> &y_true, vector<D> &y_pred, bool print_ = true){
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

	D precision(vector<D> &y_true, vector<D> &y_pred, bool print_ = true){
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

	D f1_score(vector<D> &y_true, vector<D> &y_pred, bool print_ = true){
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

	void classification_metrics(vector<D> &y_true, vector<D> &y_pred){
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
	D mean_absolute_error(vector<D> &y_true, vector<D> &y_pred, bool print_ = true){
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

	D mean_squared_error(vector<D> &y_true, vector<D> &y_pred, bool print_ = true){
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

	D root_mean_squared_error(vector<D> &y_true, vector<D> &y_pred, bool print_ = true){
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

}metrics;

struct Activation{
	static D linear(const D &t){
		return t;
	}

	static D ReLu(const D &t){
		if(t<0) return 0;
		return t;
	}

	static D sigmoid(const D &t){
		return 1/(1+expl(-t));
	}

	static D deriv_linear(const D &t){
		return 1;
	}

	static D deriv_ReLu(const D &t){
		if(t<=0) return 0;
		return 1;
	}

	static D deriv_sigmoid(const D &t){
		D sig=sigmoid(t);
		return sig*(1-sig);
	}
};

class Loss{ // Computes loss
	private:
		static constexpr D eps = 1e-15;
	public:
		static D MeanSquaredError(vector<D> &y,vector<D> &a){
			int m = y.size();
			D loss=0;
			for(int i=0;i<m;i++) loss+=(a[i]-y[i])*(a[i]-y[i]);
			loss /= m;
			return loss;
		}
		
		static D BinaryCrossentropy(vector<D> &y,vector<D> &a){
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
};

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

class Layer : Activation{
	private:
		// Given
		string name_;         		      // Name of Layer (printed in model.summary()
		int units_;               		  // # of Neurons in the layer
		string type;                	  // Layer type (Dense/...)
		function<D(const D)> activation_; // Activation function chosen for this layer
		function<D(const D)> deriv_act;   // Derivative of activation layer

		// To find
		vector<vector<D>> weight;		  // Set of weights in a single layer
		vector<D> bias;	  				  // Bias for a single layer
		vector<vector<D>> z;              // z[l] = w[l]*a[l-1] + b[l]

	public:
		string activation_name;           // Name of activation function

		Layer():units_(0){};

		Layer(const int units__,const string activation__,const string name__,const string type_):units_(units__),name_(name__),
																			type(type_),activation_name(lower_case(activation__)){
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

		Layer Dense(int units__, string activation__="", string name__=""){
			if(units>0) units__=units;
			if(activation != "") activation__=activation;
			if(name != "") name__=name;

			// Reset global variables
			units	   = 0;
			name  	   = "";
			activation = "";
			return Layer(units__,activation__,name__,"Dense");
		}

		vector<vector<D>> operator()(vector<vector<D>> &x,bool z_store = false){
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

		void element_wise_product(vector<vector<D>> &dJ_dz){
			int n = dJ_dz.size(), m = dJ_dz[0].size();
			for(int i=0;i<n;i++)
				for(int j=0;j<m;j++)
					dJ_dz[i][j] *= deriv_act(z[i][j]);
		}

		void update_weights(vector<vector<D>> &dJ_dw){
			int p = weight.size(),q = weight[0].size();
			for(int i=0;i<p;i++)
				for(int j=0;j<q;j++)
					weight[i][j] -= dJ_dw[i][j];
		}

		void update_bias(vector<D> &dJ_db){
			int n = weight.size();
			for(int i=0;i<n;i++)
				bias[i] -= dJ_db[i];
		}

		int info(int prev_units){
			string line = name_ + '(' + type + ")       (," + to_string(units_) + ")          " + to_string(units_*(1+prev_units));
			print(line);
			return units_*(1+prev_units);
		}

		string get_name(){
			return name_;
		}

		void set_name(const string name__){
			name_ = name__;
		}

		int get_units(){
			return units_;
		}

		vector<vector<D>> get_weights(){
			return weight;
		}

		void set_weights(vector<vector<D>> new_weight){
			swap(weight,new_weight);
		}

		vector<D> get_bias(){
			return bias;
		}

		void set_bias(vector<D> new_bias){
			swap(bias,new_bias);
		}
		
}layers;

struct History{
	vector<int> epoch;
	map<string,vector<D>> history;
	map<string,int> params;
};

// Keyword for model.Sequential()

int input_param = 0; // No. of features in input

// Keyword arguments for model.compile()

string loss = "";
double learning_rate = 0;

// Keyword arguments for model.fit()

int epochs = 0;
int batch_size = 32;
int steps_per_epoch = 0;
bool Shuffle = true;
pair<vector<vector<D>>,vector<D>> validation_data;						// Used for computing cross_validation loss

void reset_fit(){   // Resets global variable values
	epochs = 0;
	batch_size = 32;
	steps_per_epoch = 0;
	Shuffle = true;
	validation_data = {{},{}};
}

// Set of one or more layers
class Model : Loss{
	private:
		// Fixed
		const int print_width = 50;                                             // Printing dashed lines for summary()

		// Given
		string loss_name = "";
		function<D(vector<D> &,vector<D> &)> loss_func;						    // Loss function used in the model
		D Learning_rate;		                                                // Learning_rate for gradient descent
		int input_features;                                                     // Used for matching dimensions of input
		vector<Layer> layers;	                                             	// layers in the model

	public:
		Model():layers(),Learning_rate(0),input_features(0){};

		void add(Layer new_layer){
			if(new_layer.get_units()>0) layers.push_back(new_layer);
			else{
				string line = "Layer should have at least 1 unit.";
				print(line);
			}
		}

		Model Sequential(int input_features_ = 0, vector<Layer> layers_={}){	// input_features = 0 to prevent any initialization
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

		void summary(){
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

		template<typename... Args>
		void compile(Args&&...){
			cout<<endl;

			// Box for model.compile()
			print_header("model.compile()");

			// Box for output of model.compile()
			print_top();

			loss_name = lower_case(loss);
			Learning_rate = learning_rate;

			if(loss_name == "meansquarederror" || loss_name == "mse"){
				loss_name = "MeanSquaredError";
				loss_func = MeanSquaredError;
			}
			else if(loss_name == "binarycrossentropy" || loss_name == "bce"){
				loss_name = "BinaryCrossentropy";
				loss_func = BinaryCrossentropy;
			}
			else{
				string line = "Default loss function - MeanSquaredError will be used.";
				print(line);

				loss_name = "MeanSquaredError";
				loss_func = MeanSquaredError;
			}

			print_bottom();
			cout<<endl;

			loss = "";
			learning_rate = 0;
		}

		template<typename... Args>
		History fit(vector<vector<D>> &x_train, vector<D> &y_train, Args&&...){
			cout<<endl;
			History history;

			string line;

			// Box for model.fit()
			print_header("model.fit()");

			// Box for output of model.fit()
			print_top();

			if(batch_size <= 0) batch_size = 32; 	// Default value
			if(steps_per_epoch <= 0) steps_per_epoch = (x_train.size() + batch_size - 1)/batch_size; // Default value

			// X_train and y_train compatibility

			if(x_train.size() != y_train.size()){
				line = "No. of samples in input and output matrix are different.";
				print(line);
				print_bottom();
				cout<<endl;
				reset_fit();
				return history;
			}
			else if(layers.back().get_units() != 1){
				line = "Last layer of the model has to contain 1 unit only.";
				print(line);
				print_bottom();
				cout<<endl;
				reset_fit();
				return history;
			}
			else if(input_features != x_train[0].size()){
				line = "Expected dimension - (," + to_string(input_features) + ",)";
				print(line);
				line = "Found - (," + to_string(x_train[0].size()) + ",)";
				print(line);
				print_bottom();
				cout<<endl;
				reset_fit();
				return history;
			}
			else if(input_features == 0) input_features = x_train[0].size();

			// Validation_data compatibility
			if(!validation_data.first.empty()){
				if(validation_data.first.size() != validation_data.second.size()){
					line = "No. of samples in input and output validation_data are different.";
					print(line);
					print_bottom();
					cout<<endl;
					reset_fit();
					return history;
				}
				else if(input_features != validation_data.first[0].size()){
					line = "Expected input validation_data dimension - (," + to_string(input_features) + ",)";
					print(line);
					line = "Found - (," + to_string(validation_data.first[0].size()) + ",)";
					print(line);
					print_bottom();
					cout<<endl;
					reset_fit();
					return history;
				}
			}

			int l = layers.size();			// No. of layers
			int m = y_train.size();         // No. of samples/inputs

			history.params["epochs"] = epochs;
			history.params["steps"]  = steps_per_epoch;
			vector<D> cur_loss_vec,val_loss_vec;

			line = "Steps per epoch:-  " + to_string(steps_per_epoch);
			print(line);
			print(line = "");

			random_device rd;
		    mt19937 g(rd());
			int permutation[m];
			for(int i=0;i<m;i++) permutation[i] = i;

			auto start_time=chrono::high_resolution_clock::now();

			for(int ep=1;ep<=epochs;ep++){
				// Generate a random permutation
			    if(Shuffle) shuffle(permutation,permutation+m,g);

			    D cur_loss = 0;
			    int index = 0;  // The starting datapoint_index of a batch


   				auto t1=chrono::high_resolution_clock::now(); // Measures time per epoch
			    for(int steps=1;steps<=steps_per_epoch;steps++){
					vector<vector<D>> dJ_dz;
					vector<vector<D>> a[l+1];		// Matrix of outputs for all layers
					int m1 = batch_size;
			    	vector<D> Yt(m1);
					for(int i=0;i<m1;i++,index++){
						if(index==m) index=0;
						int ind = permutation[index];
						a[0].push_back(x_train[ind]);
						Yt[i]=y_train[ind];
					}

					a[0] = transpose(a[0]);

					// Finding output for all layers
					for(int i=0;i<l;i++)
						a[i+1]=layers[i](a[i],true);

					vector<D> a_l = a[l][0];

					// Loss after each step
					cur_loss += loss_func(Yt,a_l);

					// dJ_dz for last layer
					vector<D> v(m1);
					for(int i=0;i<m1;i++)					  	// Dividing by m so that we don't have do this in further steps
						v[i] = (a_l[i]-Yt[i])*Learning_rate/m1; // Multiplying by Learning_rate because we have to multiply before update later on.
					dJ_dz = {v};

					if(loss_name == "BinaryCrossentropy"){
						if(layers[l-1].activation_name == "sigmoid");
						else{
							;
						}
					}
					else if(loss_name == "MeanSquaredError"){
						for(D &x: dJ_dz[0]) x *= 2;
						layers[l-1].element_wise_product(dJ_dz);
					}

					// Backprop derivative calculation
					for(int i=l-1;i>=0;i--){
						vector<vector<D>> a_T = transpose(a[i]);
						vector<vector<D>> dJ_dw = multiply(dJ_dz,a_T);
						vector<D> dJ_db;
						for(vector<D> &vec:dJ_dz){
							D sum = 0;
							for(D &a:vec) sum += a;
							dJ_db.push_back(sum);
						}

						// Recursively find dJ_dz[i] from dJ_dz[i+1]
						if(i){
							vector<vector<D>> weight_T = layers[i].get_weights();
							weight_T = transpose(weight_T);
							dJ_dz = multiply(weight_T,dJ_dz);
							layers[i-1].element_wise_product(dJ_dz);
						}

						layers[i].update_weights(dJ_dw);
						layers[i].update_bias(dJ_db);

					}

				}

				cur_loss /= steps_per_epoch;

				cur_loss_vec.push_back(cur_loss);

				// Calculating validation loss
				D validation_loss = 0;
				if(!validation_data.first.empty()){
					vector<vector<D>> y_test_pred = transpose(validation_data.first);
					for(int i=0;i<l;i++)
						y_test_pred=layers[i](y_test_pred);

					validation_loss = loss_func(validation_data.second,y_test_pred[0]);

					val_loss_vec.push_back(validation_loss);
				}

				history.epoch.push_back(ep-1);

			    auto t2=chrono::high_resolution_clock::now();
				line = "Epochs = " + to_string(ep) + "/" + to_string(epochs) + "          loss = " + to_string(cur_loss);
				if(!validation_data.first.empty())
					line += "          val_loss = " + to_string(validation_loss);

				line += "         Time for epoch = " + to_string(chrono::duration_cast<chrono::milliseconds>(t2 - t1).count()) + " ms ";
				print(line);
			}

			swap(history.history["loss"],cur_loss_vec);
			if(!val_loss_vec.empty())
				swap(history.history["val_loss"],val_loss_vec);

			print(line = "");
			auto end_time=chrono::high_resolution_clock::now();
			line = "Total time = " + to_string(chrono::duration_cast<chrono::microseconds>(end_time - start_time).count()/1e6) + " s ";
			print(line);

			print_bottom();
			cout<<endl;
			reset_fit();

			return history;
		}

		vector<D> predict(vector<vector<D>> x, bool print_ = true){
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
		
		evaluate(vector<vector<D>> X_test, vector<D> y_test, bool print_ = true){
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

		void set_features(int input_features_){
			input_features = input_features_;
		}

		Layer& get_layer(string name_){
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

		vector<Layer> get_layers(){
			return layers;
		}
		
}models;

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

// Keyword arguments for read_csv()

/*
dummy_replace is can be used to format .csv while reading
by replacing certain strings of a certain header by a numeric.
E.g before using read_csv() do:
dummy_replace["Gender"]["male"] = 1;
dummy_replace["Gender"]["female"] = 0;
read_csv()

This will replace male under Gender column with 1 and female by 0.
If this is not done, "Gender" column will be ommitted to be used further in the model.
NOTE:- Once a column is included in dummy_replace all its occurences (even if it is numeric) should be mentioned in dummy_replace.
	   Otherwise, it is set to NAN, and can be later identified by - isnan(data[i][j])

dummy_replace is cleared() after each read_csv use.
*/

map<string,map<string,D>> dummy_replace;
vector<string> null_values; // Strings which are treated as NULL;

template<typename... Args>
vector<vector<D>> read_csv(const string path, bool header = true, Args&&...){
	cout<<endl;
	print_header("read_csv");

	print_top();

	ifstream file(path);
    string line,value;

    vector<vector<D>> data;             // Final output matrix
	vector<string> headers;				// String of header for columns
	vector<map<string,D>> replace;		// Efficient replace than using maps to identify column headers
	vector<bool> numeric;    			// determines if a certain column is float or not

	if(header == true){
		getline(file,line);
        stringstream ss(line);

		while(getline(ss, value, ',')){
			headers.push_back(value);
			replace.push_back(dummy_replace[value]);
		}
	}

		getline(file,line);
	    stringstream ss(line);
		vector<D> row;

		for(int col=0; getline(ss, value, ','); col++){
			if((header == true) && (!replace[col].empty())){
				numeric.push_back(true);
				if(replace[col].find(value) == replace[col].end())
					row.push_back(NAN);
				else
					row.push_back(replace[col][value]);
			}
			else{
				if(find(null_values.begin(), null_values.end(), value) == null_values.end()){
					try{
						row.push_back(stod(value));
						numeric.push_back(true);
					}
					catch(const exception& e){
						numeric.push_back(false);
					}
				}
				else{
					row.push_back(NAN);
					numeric.push_back(true);
				}
			}
		}

		data.push_back(row);

	if(header == false)
		replace.resize(numeric.size(),dummy_replace["cjqnorvby"]);

    while(getline(file, line)){
        stringstream ss(line);

		for(int col=0,c=0; getline(ss, value, ','); col++){
			if(numeric[col] == true){
				if(replace[col].empty()){
					if(find(null_values.begin(), null_values.end(), value) == null_values.end())
						row[c++] = stod(value);
					else
						row[c++] = NAN;
				}
				else
					row[c++] = replace[col][value];
			}
		}

		data.push_back(row);
    }

    print(line = "Included Columns respectively are:");
    line = "[";
    for(int i=0;i<headers.size();i++)
    	if(numeric[i] == 1)
			line += "'" + headers[i] + "',";
	line.pop_back();
	if(!line.empty()){
		line.pop_back();
		line.push_back(']');
	}
	print(line);
	print(line = "");
	print(line = "Discarded Columns are:");
	line = "[";
    for(int i=0;i<headers.size();i++)
    	if(numeric[i] == 0)
			line += "'" + headers[i] + "',";
	line.pop_back();
	if(!line.empty()){
		line.pop_back();
		line.push_back(']');
	}
	print(line);

	print_bottom();
	cout<<endl;

	dummy_replace.clear();
	null_values = {""};

    return data;
}

#pragma GCC optimize("O3")
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>
#include <random>
using namespace std;

// For printing boxes
#define VERT         (char)186
#define HORIZ        (char)205
#define TOP_LEFT     (char)201
#define TOP_RIGHT    (char)187
#define BOTTOM_LEFT  (char)200
#define BOTTOM_RIGHT (char)188
#define SCREEN_WIDTH 130

// long double -- precision of 15-33 decimal places -- But Slower -- 8 to 16 bytes
// double -- precision upto 15 decimal places -- Medium -- 8 bytes
// float -- precision upto 6 decimal places -- Relatively fast -- 4 bytes
using D = double;

string lower_case(const string &s);

void print(const vector<D> &vec);
void print(const vector<vector<D>> &vec);
void print(string &line, int width = SCREEN_WIDTH);
void print_top(int width = SCREEN_WIDTH);
void print_bottom(int width = SCREEN_WIDTH);
void print_header(string line);

void shape(vector<vector<D>> &M);

vector<vector<D>> transpose(vector<vector<D>> &M);
vector<vector<D>> multiply(vector<vector<D>> &a, vector<vector<D>> &b);

class Metric{
	public:
		// Error metrics for classification
		D accuracy(vector<D> &y_true, vector<D> &y_pred, bool print_ = true);
		D recall(vector<D> &y_true, vector<D> &y_pred, bool print_ = true);
		D precision(vector<D> &y_true, vector<D> &y_pred, bool print_ = true);
		D f1_score(vector<D> &y_true, vector<D> &y_pred, bool print_ = true);
		void classification_metrics(vector<D> &y_true, vector<D> &y_pred);
		
		// Error metrics for regression
		D mean_absolute_error(vector<D> &y_true, vector<D> &y_pred, bool print_ = true);
		D mean_squared_error(vector<D> &y_true, vector<D> &y_pred, bool print_ = true);
		D root_mean_squared_error(vector<D> &y_true, vector<D> &y_pred, bool print_ = true);
//		D (vector<D> &y_true, vector<D> &y_pred, bool print_ = true);
};

extern Metric metrics;

class Activation{
	public:
		static D linear(const D &t);
		static D ReLu(const D &t);
		static D sigmoid(const D &t);

		static D deriv_linear(const D &t);
		static D deriv_ReLu(const D &t);
		static D deriv_sigmoid(const D &t);
};

class Loss{ // Computes loss
	private:
		static constexpr D eps = 1e-15;
	public:
		static D MeanSquaredError(vector<D> &y,vector<D> &a);
		static D BinaryCrossentropy(vector<D> &y,vector<D> &a);
};

// Keyword arguments for layers.Dense()

extern int units;
extern string activation;
extern string name;

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

		Layer();

		Layer(const int units__,const string activation__,const string name__,const string type_);

		Layer Dense(int units__, string activation__="", string name__="");

		vector<vector<D>> operator()(vector<vector<D>> &x,bool z_store = false);

		void element_wise_product(vector<vector<D>> &dJ_dz);

		void update_weights(vector<vector<D>> &dJ_dw);

		void update_bias(vector<D> &dJ_db);

		int info(int prev_units);

		string get_name();

		void set_name(const string name__);

		int get_units();

		vector<vector<D>> get_weights();

		void set_weights(vector<vector<D>> new_weight);

		vector<D> get_bias();

		void set_bias(vector<D> new_bias);
};

extern Layer layers;

// Keyword for model.Sequential()

extern int input_param; // No. of features in input

// Keyword arguments for model.compile()

extern string loss;
extern double learning_rate;

// Keyword arguments for model.fit()

extern int epochs;
extern int batch_size;
extern int steps_per_epoch;
extern bool Shuffle;

void reset_fit();

// Set of one or more layers
class Model : Loss{
	private:
		// Fixed
		const int print_width = 50;                                              // Printing dashed lines for summary()

		// Given
		string loss_name = "";
		function<D(vector<D> &,vector<D> &)> loss_func;								 // Loss function used in the model
		D Learning_rate;		                                                 // Learning_rate for gradient descent
		int input_features;                                                      // Used for matching dimensions of input
		vector<Layer> layers;	                                             	 // layers in the model

	public:
		Model();

		void add(Layer new_layer);

		Model Sequential(int input_features_ = 0, vector<Layer> layers_={});		 // input_features = 0 to prevent any initialization

		void summary();

		template<typename... Args>
		void compile(Args&&...){
			cout<<endl;

			// Box for model.compile()
			print_header("model.compile()");

			// Box for output of model.compile()
			print_top();

			loss_name = lower_case(loss);
			Learning_rate = learning_rate;

			if(loss_name == "meansquarederror"){
				loss_name = "MeanSquaredError";
				loss_func = MeanSquaredError;
			}
			else if(loss_name == "binarycrossentropy"){
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
		void fit(vector<vector<D>> &x_train, vector<D> &y_train, Args&&...){
			cout<<endl;

			string line;

			// Box for model.fit()
			print_header("model.fit()");

			// Box for output of model.fit()
			print_top();

			if(batch_size <= 0) batch_size = 32; 	// Default value
			if(steps_per_epoch <= 0) steps_per_epoch = (x_train.size() + batch_size - 1)/batch_size; // Default value

			if(x_train.size() != y_train.size()){
				line = "No. of samples in input and output matrix are different.";
				print(line);
				print_bottom();
				cout<<endl;
				reset_fit();
				return;
			}
			else if(layers.back().get_units() != 1){
				line = "Last layer of the model has to contain 1 unit only.";
				print(line);
				print_bottom();
				cout<<endl;
				reset_fit();
				return;
			}
			else if(input_features != x_train[0].size()){
				line = "Expected dimension - (," + to_string(input_features) + ",)";
				print(line);
				line = "Found - (," + to_string(x_train[0].size()) + ",)";
				print(line);
				print_bottom();
				cout<<endl;
				reset_fit();
				return;
			}
			else if(input_features == 0) input_features = x_train[0].size();

			int l = layers.size();			// No. of layers
			int m = y_train.size();         // No. of samples/inputs

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

			    auto t2=chrono::high_resolution_clock::now();
				line = "Epochs = " + to_string(ep) + "/" + to_string(epochs) + "          Loss = " + to_string(cur_loss) +  "         Time for epoch = "
					 + to_string(chrono::duration_cast<chrono::milliseconds>(t2 - t1).count()) + " ms ";
				print(line);
			}

			print(line = "");
			auto end_time=chrono::high_resolution_clock::now();
			line = "Total time = " + to_string(chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count()) + " ms ";
			print(line);

			print_bottom();
			cout<<endl;
			reset_fit();
		}

		vector<D> predict(const vector<vector<D>> x);

		void set_features(int input_features_);

		Layer& get_layer(string name_);

		vector<Layer> get_layers();
};

extern Model keras;

void print(Layer &l);
void print(Model &m);

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

extern map<string,map<string,D>> dummy_replace;
extern vector<string> null_values; // Strings which are treated as NULL;

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

#include "Headers/basic.cpp"
#include <map>
#include <algorithm>
#include <functional>
#include <chrono>
#include <random>

// long double -- precision of 15-33 decimal places -- But Slower -- 8 to 16 bytes
// double -- precision upto 15 decimal places -- Medium -- 8 bytes
// float -- precision upto 6 decimal places -- Relatively fast -- 4 bytes

struct Metric{
	// Error metrics for classification
	D accuracy(vector<D> &y_true, vector<D> &y_pred, bool print_ = true);
	D recall(vector<D> &y_true, vector<D> &y_pred, bool print_ = true);
	D precision(vector<D> &y_true, vector<D> &y_pred, bool print_ = true);
	D f1_score(vector<D> &y_true, vector<D> &y_pred, bool print_ = true);
	void classification_metrics(vector<D> &y_true, vector<D> &y_pred);
	vector<vector<int>> confusion_matrix(vector<D> &y_true, vector<D> &y_pred, bool _print = true);
	
	// Error metrics for regression
	D mean_absolute_error(vector<D> &y_true, vector<D> &y_pred, bool print_ = true);
	D mean_squared_error(vector<D> &y_true, vector<D> &y_pred, bool print_ = true);
	D root_mean_squared_error(vector<D> &y_true, vector<D> &y_pred, bool print_ = true);
		
};

extern Metric metrics;

struct Activation{
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

		void update_weights(vector<vector<D>> &dJ_dw, double Learning_rate);

		void update_bias(vector<D> &dJ_db, double Learning_rate);

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

extern string monitor;
extern string mode;
extern int patience;

class Callback{
	private:
		string Mode;
		int Patience;
		D best_metric;
		int best_epoch;
		int cur_epoch;
		string Type;

		void reset();

		Callback(string monitor_, string mode_, int patience_, string type_);
		
	public:
		string Monitor;
	
		template<typename... Args>
		static Callback EarlyStopping(Args&&...){
			return Callback(lower_case(monitor), lower_case(mode), patience, "EarlyStopping");
		}

		bool should_stop(D cur_metric);
};

struct History{
	vector<int> epoch;
	map<string,vector<D>> history;
	map<string,int> params;
};

// Keyword for model.Sequential()

extern int input_param; // No. of features in input

// Keyword arguments for model.compile()

extern string loss;
extern string optimizer;
extern double learning_rate, beta_1, beta_2;
extern double epsilon;

// Keyword arguments for model.fit()

extern int epochs;
extern int batch_size;
extern int steps_per_epoch;
extern bool Shuffle;
extern pair<vector<vector<D>>,vector<D>> validation_data;						// Used for computing cross_validation loss
extern vector<Callback> callbacks;

void reset_fit();

// Set of one or more layers
class Model : Loss{
	private:
		// Fixed
		const int print_width = 50;                                             // Printing dashed lines for summary()

		// Given
		string loss_name = "";
		function<D(vector<D> &,vector<D> &)> loss_func;						    // Loss function used in the model
		string Optimizer;                                                       // name of optimizer
		double Learning_rate;		                                            // Learning_rate for gradient descent
		double Beta_1, Beta_2;                                                  // hyperparameters
		double Epsilon;                                                         // Numerical stability
		int input_features;                                                     // Used for matching dimensions of input
		vector<Layer> layers;	                                             	// layers in the model
		
	public:
		Model();

		void add(Layer new_layer);

		Model Sequential(int input_features_ = 0, vector<Layer> layers_={});	// input_features = 0 to prevent any initialization

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
			Beta_1 = beta_1;
			Beta_2 = beta_2;
			Epsilon = epsilon;

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
			
			Optimizer = lower_case(optimizer);
			if(Optimizer != "adam")
				Optimizer = "sgd";

			print_bottom();
			cout<<endl;

			loss = "";
			optimizer = "";
			learning_rate = 0.001;
			beta_1 = 0.9;
			beta_2 = 0.999;
			epsilon = 1e-7;
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
			
			vector<vector<D>> m_w[l], v_w[l];
			vector<D> m_b[l], v_b[l];
			double Beta_1t = 1, Beta_2t = 1;

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
					for(int i=0;i<m1;i++)					  
						v[i] = (a_l[i]-Yt[i])/m1;   // Dividing by m1 so that we don't have to do it in later steps
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

					if(Optimizer == "adam"){
						Beta_1t *= Beta_1;
						Beta_2t *= Beta_2;
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

						if(Optimizer == "adam"){
							int n = dJ_dw.size(), m = dJ_dw[0].size();
							if(m_b[i].empty()){
								m_w[i] = multiply(dJ_dw, 1-Beta_1);
								v_w[i] = multiply(hadamard_product(dJ_dw,dJ_dw), 1-Beta_2);
								m_b[i] = multiply(dJ_db, 1-Beta_1);
								v_b[i] = multiply(hadamard_product(dJ_db,dJ_db), 1-Beta_2);
							}
							else{
								for(int p=0;p<n;p++){
									for(int q=0;q<m;q++){
										m_w[i][p][q] = Beta_1*m_w[i][p][q] + (1-Beta_1)*dJ_dw[p][q];
										v_w[i][p][q] = Beta_2*v_w[i][p][q] + (1-Beta_2)*dJ_dw[p][q]*dJ_dw[p][q];
									}
									m_b[i][p] = Beta_1*m_b[i][p] + (1-Beta_1)*dJ_db[p];
									v_b[i][p] = Beta_2*v_b[i][p] + (1-Beta_2)*dJ_db[p]*dJ_db[p];
								}
							}
							
							for(int p=0;p<n;p++){
								for(int q=0;q<m;q++)
									dJ_dw[p][q] = m_w[i][p][q]/((1-Beta_1t)*(sqrtl(v_w[i][p][q]/(1-Beta_2t))+Epsilon));
								dJ_db[p] = m_b[i][p]/((1-Beta_1t)*(sqrtl(v_b[i][p]/(1-Beta_2t))+Epsilon));
							}
						}

						layers[i].update_weights(dJ_dw, Learning_rate);
						layers[i].update_bias(dJ_db, Learning_rate);

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
				
				bool stop = false;
				for(Callback &callback:callbacks){
					if(callback.Monitor == "val_loss"){
						if(callback.should_stop(validation_loss)){
							print(line = "");
							print(line = "Callback: Early Stopping due to val_loss");
							stop = true;
							break;
						}
					}
					else if(callback.Monitor == "loss"){
						if(callback.should_stop(cur_loss)){
							print(line = "");
							print(line = "Callback: Early Stopping due to loss");
							stop = true;
							break;
						}
					}
				}
				if(stop)
					break;
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
		
		vector<D> predict(vector<vector<D>> x, bool print = true);

		D evaluate(vector<vector<D>> X_test, vector<D> y_test, bool print_ = true);

		void set_features(int input_features_);

		Layer& get_layer(string name_);

		vector<Layer> get_layers();
};

extern Model models;

void print(Layer &l);
void print(Model &m);

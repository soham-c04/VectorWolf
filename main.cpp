#include "VectorWolf.h"

#include <iomanip>
using namespace std;

/*
Below is an example to demonstrate functioning of VectorWolf library and determining ideal conditions for CoffeeRoasting.

Example from source - https://github.com/AmirhosseinKoochakian2003/Machine-Learning-Specialization-Coursera/blob/master/Advanced%20Learning
					  %20Algorithms/Week%201/W1%20Labs/C2_W1_Lab02_CoffeeRoasting_TF.ipynb
Data from source - https://github.com/HoomKh/Coffee-Roasting-Deeplearning/blob/main/CoffeeRosting-Deeplearning.ipynb
*/

int main(){
	vector<vector<D>> X = read_csv("Kaggle/CoffeeRoasting/input.csv");
	vector<vector<D>> Y = read_csv("Kaggle/CoffeeRoasting/output.csv");
	shape(X);
	shape(Y);
	int m = X.size();

	int tile = 1000;
	vector<vector<D>> Xt;
	vector<D> Yt;
	while(tile--){
		for(int i=0;i<m;i++){
			Yt.push_back(Y[i][0]);
			Xt.push_back(X[i]);
		}
	}

	shape(Xt);

	Model model=keras.Sequential(
		input_param = 2,
		{
		layers.Dense(2, activation="linear", name = "Layer1"),
		layers.Dense(2, activation="sigmoid"),
        layers.Dense(1, activation="sigmoid", name = "Layer3"),
        layers.Dense(-1),
        layers.Dense(0,"ReLu")
		}
	);

	model.summary();

	model.compile(loss = "BinaryCrossentropy", learning_rate = 0.1);

	vector<vector<D>>
		W1 = {
	    {-1.0, 5.0},
	    {10.0, 7.0}},
		W2 = {
	    {-15.0, -1.0},
		{10.0, 7.0}},
		W3 = {
	    {-12.0},
	    {-15.0}};

	vector<D>
		b1 = {-41.0, 22.0},
		b2 = {-28.0, -91.0},
		b3 = {31.0};

	// Update weights to set value
	
	model.get_layer("Layer1").set_weights(transpose(W1));
	model.get_layer("Layer2").set_weights(transpose(W2));
	model.get_layer("Layer3").set_weights(transpose(W3));

	model.get_layer("Layer1").set_bias(b1);
	model.get_layer("Layer2").set_bias(b2);
	model.get_layer("Layer3").set_bias(b3);

	// Start Gradient Descent
	
	cout << fixed << setprecision(7);
	
	model.fit(
	    Xt,Yt,
	    epochs = 10,
	    batch_size = 25000,
//	    steps_per_epoch = 8,
	    Shuffle = false
	);

	print(model);
	
	vector<D> Y_predict = model.predict(Xt);
	
	m = Yt.size();
	D accuracy = 0;
	for(int i=0;i<m;i++)
		if((Y_predict[i] >= 0.5) == Yt[i])
			accuracy += 100;
			
	accuracy /= m;
	cout << "\nAccuracy: "<< accuracy << endl << endl;

	return 0;
}

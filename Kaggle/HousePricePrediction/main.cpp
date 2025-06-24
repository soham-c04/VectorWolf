//#pragma GCC optimize("O3")

#include "VectorWolf.cpp"
#include "Headers/File_IO.cpp"

//#include "VectorWolf_all.cpp"

int main(){
	vector<vector<D>> X_train = read_csv("Kaggle/HousePricePrediction/X_train.csv", false),
					  X_test = read_csv("Kaggle/HousePricePrediction/X_test.csv", false),
					  y_train_ = read_csv("Kaggle/HousePricePrediction/y_train.csv", false),
			  		  y_test_ = read_csv("Kaggle/HousePricePrediction/y_test.csv", false);

	vector<D> y_train = transpose(y_train_)[0],
			  y_test  = transpose(y_test_)[0];

	shape(X_train);
	shape(X_test);

	Model model = models.Sequential(input_param = X_train[0].size());

	model.add(layers.Dense(19, activation="relu"));
	model.add(layers.Dense(19, activation="relu"));
	model.add(layers.Dense(19, activation="relu"));
	model.add(layers.Dense(19, activation="relu"));
	model.add(layers.Dense(1 , activation="relu"));

	model.summary();

	model.compile(loss = "mse", learning_rate = 0.001);

	History history = model.fit(
	    X_train,y_train,
	    epochs = 50,
	    batch_size = 20,
	    validation_data = {X_test, y_test},
	    Shuffle = false
	);
	
	write_csv("Kaggle/HousePricePrediction/loss.csv", history.history["loss"]);
	write_csv("Kaggle/HousePricePrediction/val_loss.csv", history.history["val_loss"]);
	
	vector<D> prediction = model.predict(X_test);

	model.evaluate(X_test, y_test);

	metrics.mean_absolute_error(y_test, prediction);

	metrics.root_mean_squared_error(y_test, prediction);

	return 0;
}

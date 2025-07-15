//#pragma GCC optimize("O3")

#include "VectorWolf.cpp"
#include "Headers/File_IO.cpp"

//#include "VectorWolf_all.cpp"
int main(){
	vector<vector<D>> X_train = read_csv("Kaggle/CancerPrediction/X_train.csv", false),
					  X_test = read_csv("Kaggle/CancerPrediction/X_test.csv", false),
					  y_train_ = read_csv("Kaggle/CancerPrediction/y_train.csv", false),
			  		  y_test_ = read_csv("Kaggle/CancerPrediction/y_test.csv", false);

	vector<D> y_train = transpose(y_train_)[0],
			  y_test  = transpose(y_test_)[0];

	shape(X_train);
	shape(X_test);

	Model model = models.Sequential(input_param = X_train[0].size());

	model.add(layers.Dense(units = 30, activation="relu"));
	model.add(layers.Dense(units = 15, activation="relu"));
	model.add(layers.Dense(units = 1,  activation="sigmoid"));

	model.summary();

	model.compile(loss = "bce", optimizer = "adam");

	History history = model.fit(
	    X_train,y_train,
	    epochs = 500,
	    batch_size = 20,
	    validation_data = {X_test, y_test},
	    callbacks = {Callback::EarlyStopping(monitor="val_loss", mode="min", patience=25)},
	    Shuffle = false
	);
	
	write_csv("Kaggle/CancerPrediction/loss.csv", history.history["loss"]);
	write_csv("Kaggle/CancerPrediction/val_loss.csv", history.history["val_loss"]);
	
	vector<D> prediction = model.predict(X_test);
	
	metrics.classification_metrics(y_test, prediction);

	metrics.confusion_matrix(y_test, prediction);

	return 0;
}

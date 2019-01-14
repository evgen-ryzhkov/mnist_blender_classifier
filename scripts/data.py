import mnist
import numpy as np

class MNISTData:
	# MNIST data set was separated on train (first 60 0000) and test (last 10 000) sets
	# mnist lib has special methods for getting train and test sets
	def get_train_set(self):
		X_train = mnist.train_images()
		y_train = mnist.train_labels()
		mixed_indexes = np.random.permutation(60000)
		X_train, y_train = X_train[mixed_indexes], y_train[mixed_indexes]
		X_train_prepared = self._prepare_data(X_train)
		return X_train_prepared, y_train

	def get_validate_and_test_sets(self):
		X_test = mnist.test_images()
		y_test = mnist.test_labels()
		mixed_indexes = np.random.permutation(10000)
		X_test, y_test = X_test[mixed_indexes], y_test[mixed_indexes]
		X_prepared = self._prepare_data(X_test)
		X_valid = X_prepared[0:5000, :]
		y_valid = y_test[0:5000]
		X_test = X_prepared[5001:10000, :]
		y_test = y_test[5001:10000]
		return X_valid, y_valid, X_test, y_test

	def _prepare_data(self, not_prepared_data):
		# Normalizing the RGB codes by dividing it to the max RGB value.
		prepared_data = not_prepared_data / 255

		# used models needs 2 dimensions data
		prepared_data = prepared_data.reshape(prepared_data.shape[0], 784)

		# reduce RAM requirements
		prepared_data = prepared_data.astype(np.float32)

		return prepared_data


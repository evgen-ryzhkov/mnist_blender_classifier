import settings
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import time


class DigitClassifier:

	def train_1st_level_models(self, X_train, y_train):
		start_time = time.time()
		rnd_forest_clf = self._train_rnd_forest_clf(X_train, y_train, idx_str='1')
		print('\n--------------------------------------------------')
		extra_trees_clf = self._train_extra_trees_clf(X_train, y_train, idx_str='2')
		print('\n--------------------------------------------------')
		svm_clf = self._train_svm_clf(X_train, y_train, idx_str='3')
		print('\n--------------------------------------------------')
		end_time = time.time()
		print('Total train time = ', round(end_time - start_time), 's')
		# one core - 8166 s / 136.1 minutes (13% using of processor)
		# multi core - 5335 s / 88.9 minutes

	def _train_rnd_forest_clf(self, X_train, y_train, idx_str):
		print('Train RandomForestClassifier was started...')
		clf = RandomForestClassifier(n_jobs=-1)
		clf.fit(X_train, y_train)
		y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3, n_jobs=-1)
		self._calculate_model_metrics(y_train, y_train_pred, model_name='RandomForestClassifier')
		self._save_model(clf, idx_str)
		return clf

	def _train_extra_trees_clf(self, X_train, y_train, idx_str):
		print('Train ExtraTreesClassifier was started...')
		clf = ExtraTreesClassifier(n_jobs=-1)
		clf.fit(X_train, y_train)
		y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3, n_jobs=-1)
		self._calculate_model_metrics(y_train, y_train_pred, model_name='ExtraTreesClassifier')
		self._save_model(clf, idx_str)
		return clf

	def _train_svm_clf(self, X_train, y_train, idx_str):
		print('Train SVM classifier was started...')
		clf = SVC(probability=True)
		clf.fit(X_train, y_train)
		print('SVC cross_validation was started...')
		y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3, n_jobs=-1)
		self._calculate_model_metrics(y_train, y_train_pred, model_name='SVM classifier')
		self._save_model(clf, idx_str)
		return clf

	def train_2st_level_model(self, X_valid, y_valid):
		print('Creating predicted data set...')
		predicted_1st_level_models_data_set = self._create_predicted_data_set_by_1st_level_models(X_valid)
		print('Train RandomForestClassifier 2nd level...')
		clf = RandomForestClassifier(n_jobs=-1)
		clf.fit(predicted_1st_level_models_data_set, y_valid)
		y_pred = cross_val_predict(clf, predicted_1st_level_models_data_set, y_valid, cv=3, n_jobs=-1)
		self._calculate_model_metrics(y_valid, y_pred, model_name='RandomForestClassifier 2nd level')
		self._save_model(clf, 'blender')

	def compare_individual_models_and_blender(self, X_test, y_test):
		y_pred_1 = self._get_predicted_values_1st_level_model(X_test, 'clf_1.sav')
		print('Individual RandomForestClassifier accuracy =', accuracy_score(y_test, y_pred_1))

		y_pred_2 = self._get_predicted_values_1st_level_model(X_test, 'clf_2.sav')
		print('Individual ExtraTreesClassifier accuracy =', accuracy_score(y_test, y_pred_2))

		y_pred_3 = self._get_predicted_values_1st_level_model(X_test, 'clf_3.sav')
		print('Individual SVM classifier accuracy =', accuracy_score(y_test, y_pred_3))

		y_pred_1st_level_data_size = len(y_pred_1)
		y_pred_level_1 = np.zeros((y_pred_1st_level_data_size, 3), dtype=int)
		y_pred_level_1[:, 0] = y_pred_1
		y_pred_level_1[:, 1] = y_pred_2
		y_pred_level_1[:, 2] = y_pred_3

		y_pred_blender = self._get_predicted_values_1st_level_model(y_pred_level_1, 'clf_blender.sav')
		print('Blender classifier accuracy =', accuracy_score(y_test, y_pred_blender))

	def _create_predicted_data_set_by_1st_level_models(self, X_valid):
		data_set_size = X_valid.shape[0]
		predicted_1st_level_models_data_set = np.zeros((data_set_size, 3), dtype=int)

		predicted_1st_level_models_data_set[:, 0] = self._get_predicted_values_1st_level_model(X_valid, 'clf_1.sav')
		predicted_1st_level_models_data_set[:, 1] = self._get_predicted_values_1st_level_model(X_valid, 'clf_2.sav')
		predicted_1st_level_models_data_set[:, 2] = self._get_predicted_values_1st_level_model(X_valid, 'clf_3.sav')

		return predicted_1st_level_models_data_set

	@staticmethod
	def _get_predicted_values_1st_level_model(X, file_name):
		try:
			clf = joblib.load(settings.MODELS_DIR + file_name)
			return clf.predict(X)
		except FileNotFoundError:
			raise ValueError('Model file not found!')

	@staticmethod
	def _calculate_model_metrics(y_train, y_pred, model_name):
		print('Calculating metrics...')
		labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
		precision, recall, fscore, support = precision_recall_fscore_support(
			y_train, y_pred,
			labels=labels)

		precision = np.reshape(precision, (10, 1))
		recall = np.reshape(recall, (10, 1))
		fscore = np.reshape(fscore, (10, 1))
		data = np.concatenate((precision, recall, fscore), axis=1)
		df = pd.DataFrame(data)
		df.columns = ['Precision', 'Recall', 'Fscore']
		print(model_name, '\n')
		print(df)

		print('\n Average values')
		print('Precision = ', df['Precision'].mean())
		print('Recall = ', df['Recall'].mean())
		print('F1 score = ', df['Fscore'].mean())

	@staticmethod
	def _save_model(classifier, idx_str):
		try:
			joblib.dump(classifier, settings.MODELS_DIR + settings.CLF_ROOT_FILE_NAME + idx_str + settings.CLF_EXTENSION_FILE_NAME)
		except IOError:
			raise ValueError('Something wrong with file save operation.')
		except ValueError:
			raise ValueError('Something wrong with classifier or idx value.')

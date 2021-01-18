from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef
from sklearn.metrics import cohen_kappa_score,roc_auc_score,confusion_matrix,classification_report

def evalModel(y_test, y_pred):
	'''
	    Evaluate the performance of the model.
	    Args:
	       y_test: The array of features to be tested against.
	       y_pred: Model predictions.
        Returns: Accuracy, Precision, Recall, F1 score, Cohens kappa, Matthews correlation coefficient
                 of the model after evaluation.

	'''
	y_test = y_test.reshape(y_test.shape[0],y_test.shape[2])
    y_test =np.argmax(y_test,axis=1)

    # accuracy: (tp + tn) / (p + n)
	accuracy = accuracy_score(y_test, y_pred)
	print('Accuracy: %f' % accuracy)
	# precision tp / (tp + fp)
	precision = precision_score(y_test, y_pred,average='weighted')
	print('Precision: %f' % precision)
	# recall: tp / (tp + fn)
	recall = recall_score(y_test, y_pred,average='weighted')
	print('Recall: %f' % recall)
	# f1: 2 tp / (2 tp + fp + fn)
	f1 = f1_score(y_test, y_pred,average='weighted')
	print('F1 score: %f' % f1)
	 
	# kappa
	kappa = cohen_kappa_score(y_test, y_pred)
	print('Cohens kappa: %f' % kappa)
	MatthewsCorrCoef = matthews_corrcoef(y_test, y_pred)
	print('Matthews correlation coefficient: %f' % MatthewsCorrCoef)
	# ROC AUC
	'''auc = roc_auc_score(y_test, y_pred)
	print('ROC AUC: %f' % auc)'''
	# confusion matrix
	matrix = classification_report(y_test, y_pred)
	print(matrix)

	return {
	       "Accuracy": accuracy,
	       "Precision": precision,
	       "Recall": recall,
	       "F1 score": f1,
	       "Cohens kappa": kappa,
	       "Matthews correlation coefficient": MatthewsCorrCoef
	}

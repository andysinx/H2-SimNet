from utils.utils_data import read_and_print_dataset_description, plot_time_series_graphics
import anomaly_detection_classifiers.anomaly_detection_classifier as adc   
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    
    #Dataset Info
    datasets = read_and_print_dataset_description(["dataset_enea/raw_normal_dataset_def.csv", "dataset_enea/raw_anomaly1_dataset_def.csv", "dataset_enea/raw_anomaly2_dataset_def.csv", "dataset_enea/raw_anomaly3_dataset_def.csv"])
    plot_time_series_graphics(datasets)
    
    #Anomaly Detection Classifiers test
    anom_det_classifiers_obj = adc.Anomaly_Detection_Classifier(datasets)
    X_train, X_val, X_test, y_train, y_val, y_test = anom_det_classifiers_obj.split_dataset()
    
    #One class SVM classifier test:
    one_class_svm = anom_det_classifiers_obj.one_class_svm_classifier(X_train, X_val, X_test, y_train, y_val, y_test)
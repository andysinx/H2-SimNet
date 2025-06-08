import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import select_features
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, make_scorer
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection
import seaborn as sns
import warnings
import os
warnings.filterwarnings("ignore")

class Anomaly_Detection_Classifier():
    
    def __init__(self, dataset: np.ndarray): 
        self.dataset = dataset
        
    
    def __multivariate_time_series_features_extraction_and_selection(self):
        
        # Feature Extraction
        self.dataset[0].reset_index(inplace=True)
        self.dataset[0].rename(columns={'index': 'ID'}, inplace=True)
        self.dataset[1].reset_index(inplace=True)
        self.dataset[1].rename(columns={'index': 'ID'}, inplace=True)
        self.dataset[2].reset_index(inplace=True)
        self.dataset[2].rename(columns={'index': 'ID'}, inplace=True)
        self.dataset[3].reset_index(inplace=True)
        self.dataset[3].rename(columns={'index': 'ID'}, inplace=True)

        y_tot = pd.concat([self.dataset[0]['Y'],  self.dataset[1]['Y'],  self.dataset[2]['Y'],  self.dataset[3]['Y']], ignore_index=True)
        extracted_features_norm = extract_features(self.dataset[0].drop(columns='Y'), column_id="ID", column_sort="Time")

        extracted_features_anom_1 = extract_features(self.dataset[1].drop(columns='Y'), column_id="ID", column_sort="Time")

        extracted_features_anom_2 = extract_features(self.dataset[2].drop(columns='Y'), column_id="ID", column_sort="Time")

        extracted_features_anom_3 = extract_features(self.dataset[3].drop(columns='Y'), column_id="ID", column_sort="Time")

        impute(extracted_features_norm)
        impute(extracted_features_anom_1)
        impute(extracted_features_anom_2)
        impute(extracted_features_anom_3)

        extracted_features_total = pd.concat([extracted_features_norm,  extracted_features_anom_1, extracted_features_anom_2,  extracted_features_anom_3], ignore_index=True)
        print(extracted_features_total)
        print(extracted_features_total.info())
        print(extracted_features_total.describe())
        
        #Features Selection
        features_filtered = select_features(extracted_features_total, y_tot)
        print(features_filtered)
        print(features_filtered.info())
        print(features_filtered.describe())
        
        
        files = os.listdir("./data_extraction")
        empty_folder = True
        
        if files[0].lower().endswith(('.png')):
            empty_folder = False  # La cartella contiene almeno un'immagine
            print("Folder data_extraction not empty")
                
        if (empty_folder):
            features_filtered.to_excel("data_extraction/selected_features.xlsx", index=False)
            
        return features_filtered, y_tot
    
    
    def split_dataset(self):
        X, y = self.__multivariate_time_series_features_extraction_and_selection()
        X = X.values  
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    
    def one_class_svm_classifier(self, X_train, X_val, X_test, y_train, y_val, y_test):
        
        oneclass_svm = OneClassSVM()
        
        param_grid = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'nu': np.linspace(0.01, 1, 5),
            'gamma': np.linspace(0.01, 1, 5)
        }
        
        f1_scorer = make_scorer(f1_score, average='binary')
        grid_search = GridSearchCV(estimator=oneclass_svm, param_grid=param_grid, scoring=f1_scorer, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred_val = best_model.predict(X_val)
        
        opt_params = {
            'Best_Parameter': ['Best_params', 'Best_F1_score_val'],
            'Value': [grid_search.best_params_, grid_search.best_score_]
        }
        
        df_best_params = pd.DataFrame(opt_params)
        df_best_params.to_csv('result/oneclass_svm/best_params_oneclass_svm.csv', index=False)
        
        #validation_metrics
        accuracy_val = accuracy_score(y_val, y_pred_val)
        f1_score_val = f1_score(y_val, y_pred_val, pos_label=1)
        confusion_matrix_val = confusion_matrix(y_val, y_pred_val, labels=[1,-1])
       
        
        #test_metrics
        y_pred_test = best_model.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        f1_score_test = f1_score(y_test, y_pred_test, pos_label=1)
        confusion_matrix_test = confusion_matrix(y_test, y_pred_test, labels=[1,-1])
        print('accuracy_test: ',accuracy_test)
        print('f1_score_test: ',f1_score_test)
        
        metrics = {
            'Metric': ['Accuracy_val', 'F1_Score_val', 'Accuracy_test', 'F1_Score_test'],
            'Value': [accuracy_val, f1_score_val, accuracy_test, f1_score_test]
        }
        
        df_metrics = pd.DataFrame(metrics)
        df_metrics.to_csv('result/oneclass_svm/metrics_oneclass_svm.csv', index=False)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix_val, annot=True, cmap='Blues', fmt='g', xticklabels=[1, -1], yticklabels=[1, -1])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix Validation Set')
        plt.savefig('result/oneclass_svm/confusion_matrix_validation.png', dpi=300, bbox_inches='tight')
        
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix_test, annot=True, cmap='Blues', fmt='g', xticklabels=[1, -1], yticklabels=[1, -1])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix Test Set')
        plt.savefig('result/oneclass_svm/confusion_matrix_test.png', dpi=300, bbox_inches='tight')
        return oneclass_svm
    
    
    
    def isolation_forest_classifier(self, X_train, X_val, X_test, y_train, y_val, y_test):
        
        iso_forest = IsolationForest()
 
        param_grid = {
            'n_estimators': list(range(1,5)),
            'contamination': np.linspace(0.1, 0.5, 5) ,
            'max_features': np.linspace(0.1, 1.0, 5),
            'bootstrap': True,
            'random_state': 42
        }
        
        
        f1_scorer = make_scorer(f1_score, average='binary')
        grid_search = GridSearchCV(estimator=iso_forest, param_grid=param_grid, scoring=f1_scorer, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred_val = best_model.predict(X_val)
        
        opt_params = {
            'Best_Parameter': ['Best_params', 'Best_F1_score_val'],
            'Value': [grid_search.best_params_, grid_search.best_score_]
        }
        
        df_best_params = pd.DataFrame(opt_params)
        df_best_params.to_csv('result/isolation_forest/best_params_isolation_forest.csv', index=False)
        
        #validation_metrics
        accuracy_val = accuracy_score(y_val, y_pred_val)
        f1_score_val = f1_score(y_val, y_pred_val, pos_label=1)
        confusion_matrix_val = confusion_matrix(y_val, y_pred_val, labels=[1,-1])
       
        
        #test_metrics
        y_pred_test = best_model.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        f1_score_test = f1_score(y_test, y_pred_test, pos_label=1)
        confusion_matrix_test = confusion_matrix(y_test, y_pred_test, labels=[1,-1])
        print('accuracy_test: ',accuracy_test)
        print('f1_score_test: ',f1_score_test)
        
        metrics = {
            'Metric': ['Accuracy_val', 'F1_Score_val', 'Accuracy_test', 'F1_Score_test'],
            'Value': [accuracy_val, f1_score_val, accuracy_test, f1_score_test]
        }
        
        df_metrics = pd.DataFrame(metrics)
        df_metrics.to_csv('result/isolation_forest/metrics_isolation_forest.csv', index=False)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix_val, annot=True, cmap='Blues', fmt='g', xticklabels=[1, -1], yticklabels=[1, -1])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix Validation Set')
        plt.savefig('result/isolation_forest/confusion_matrix_validation.png', dpi=300, bbox_inches='tight')
        
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix_test, annot=True, cmap='Blues', fmt='g', xticklabels=[1, -1], yticklabels=[1, -1])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix Test Set')
        plt.savefig('result/isolation_forest/confusion_matrix_test.png', dpi=300, bbox_inches='tight')
        return iso_forest
     
    def random_forest_classifier(self, X_train, X_val, X_test, y_train, y_val, y_test):
        
        rand_forest = RandomForestClassifier()
 
        param_grid = {
            'n_estimators': list(range(1,5)),
            'max_depth': list(range(1,10)),
            'random_state': 42,
            'max_samples':  np.linspace(0.1, 0.5, 5),
            'bootstrap': True
        }
            
        f1_scorer = make_scorer(f1_score, average='binary')
        grid_search = GridSearchCV(estimator=rand_forest, param_grid=param_grid, scoring=f1_scorer, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred_val = best_model.predict(X_val)
        
        opt_params = {
            'Best_Parameter': ['Best_params', 'Best_F1_score_val'],
            'Value': [grid_search.best_params_, grid_search.best_score_]
        }
        
        df_best_params = pd.DataFrame(opt_params)
        df_best_params.to_csv('result/random_forest/best_params_random_forest.csv', index=False)
        
        #validation_metrics
        accuracy_val = accuracy_score(y_val, y_pred_val)
        f1_score_val = f1_score(y_val, y_pred_val, pos_label=1)
        confusion_matrix_val = confusion_matrix(y_val, y_pred_val, labels=[1,-1])
       
        
        #test_metrics
        y_pred_test = best_model.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        f1_score_test = f1_score(y_test, y_pred_test, pos_label=1)
        confusion_matrix_test = confusion_matrix(y_test, y_pred_test, labels=[1,-1])
        print('accuracy_test: ',accuracy_test)
        print('f1_score_test: ',f1_score_test)
        
        metrics = {
            'Metric': ['Accuracy_val', 'F1_Score_val', 'Accuracy_test', 'F1_Score_test'],
            'Value': [accuracy_val, f1_score_val, accuracy_test, f1_score_test]
        }
        
        df_metrics = pd.DataFrame(metrics)
        df_metrics.to_csv('result/random_forest/metrics_random_forest.csv', index=False)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix_val, annot=True, cmap='Blues', fmt='g', xticklabels=[1, -1], yticklabels=[1, -1])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix Validation Set')
        plt.savefig('result/random_forest/confusion_matrix_validation.png', dpi=300, bbox_inches='tight')
        
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix_test, annot=True, cmap='Blues', fmt='g', xticklabels=[1, -1], yticklabels=[1, -1])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix Test Set')
        plt.savefig('result/random_forest/confusion_matrix_test.png', dpi=300, bbox_inches='tight')
        return rand_forest
        
     

    def local_outlier_factor_classifier(self, X_train, X_val, X_test, y_train, y_val, y_test):
        
        loc_outl = LocalOutlierFactor()
 
        param_grid = {
            'algorithm': ['ball_tree', 'kd_tree'],
            'n_neighbors': list(range(10,15)),
            'leaf_size' : list(range(30,35)),
            'contamination': np.linspace(0.1, 0.5, 5), 
            'novelty': True
        }
        
        f1_scorer = make_scorer(f1_score, average='binary')
        grid_search = GridSearchCV(estimator=loc_outl, param_grid=param_grid, scoring=f1_scorer, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred_val = best_model.predict(X_val)
        
        opt_params = {
            'Best_Parameter': ['Best_params', 'Best_F1_score_val'],
            'Value': [grid_search.best_params_, grid_search.best_score_]
        }
        
        df_best_params = pd.DataFrame(opt_params)
        df_best_params.to_csv('result/local_out_fac/best_params_lof.csv', index=False)
        
        #validation_metrics
        accuracy_val = accuracy_score(y_val, y_pred_val)
        f1_score_val = f1_score(y_val, y_pred_val, pos_label=1)
        confusion_matrix_val = confusion_matrix(y_val, y_pred_val, labels=[1,-1])
       
        
        #test_metrics
        y_pred_test = best_model.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        f1_score_test = f1_score(y_test, y_pred_test, pos_label=1)
        confusion_matrix_test = confusion_matrix(y_test, y_pred_test, labels=[1,-1])
        print('accuracy_test: ',accuracy_test)
        print('f1_score_test: ',f1_score_test)
        
        metrics = {
            'Metric': ['Accuracy_val', 'F1_Score_val', 'Accuracy_test', 'F1_Score_test'],
            'Value': [accuracy_val, f1_score_val, accuracy_test, f1_score_test]
        }
        
        df_metrics = pd.DataFrame(metrics)
        df_metrics.to_csv('result/local_out_fac/metrics_lof.csv', index=False)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix_val, annot=True, cmap='Blues', fmt='g', xticklabels=[1, -1], yticklabels=[1, -1])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix Validation Set')
        plt.savefig('result/local_out_fac/confusion_matrix_validation.png', dpi=300, bbox_inches='tight')
        
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix_test, annot=True, cmap='Blues', fmt='g', xticklabels=[1, -1], yticklabels=[1, -1])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix Test Set')
        plt.savefig('result/local_out_fac/confusion_matrix_test.png', dpi=300, bbox_inches='tight')
        

        X_scores = loc_outl.negative_outlier_factor_
        y_pred_val = loc_outl.predict(X_val)
        n_errors = (y_pred_val != y_val).sum()
        
        
        def update_legend_marker_size(handle, orig):
            handle.update_from(orig)
            handle.set_sizes([20])
        
        plt.scatter(X_train[:, 0], X_train[:, 1], color="k", s=3.0, label="Data points")
        radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
        scatter = plt.scatter(
            X_train[:, 0],
            X_train[:, 1],
            s=1000 * radius,
            edgecolors="r",
            facecolors="none",
            label="Outlier scores",
        )
        plt.axis("tight")
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))
        plt.xlabel("prediction errors: %d" % (n_errors))
        plt.legend(
            handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
        )
        plt.title("Local Outlier Factor (LOF)")
        plt.savefig('result/local_out_fac/lof_data_outlier.png', dpi=300, bbox_inches='tight')
        return loc_outl
        
        
    def bagging(self, classifier, X_train, X_val, X_test, y_train, y_val, y_test):
           
        bagging_model = BaggingClassifier()
 
        param_grid = {
            'base_estimator': classifier,
            'n_estimators': list(range(1,8)),
            'max_samples':  np.linspace(0.1, 0.5, 5), 
            'bootstrap': True,
            'random_state':42
        }
            
        f1_scorer = make_scorer(f1_score, average='binary')
        grid_search = GridSearchCV(estimator=bagging_model, param_grid=param_grid, scoring=f1_scorer, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred_val = best_model.predict(X_val)
        
        opt_params = {
            'Best_Parameter': ['Best_params', 'Best_F1_score_val'],
            'Value': [grid_search.best_params_, grid_search.best_score_]
        }
        
        df_best_params = pd.DataFrame(opt_params)
        df_best_params.to_csv('result/bagging/best_params_lof.csv', index=False)
        
        #validation_metrics
        accuracy_val = accuracy_score(y_val, y_pred_val)
        f1_score_val = f1_score(y_val, y_pred_val, pos_label=1)
        confusion_matrix_val = confusion_matrix(y_val, y_pred_val, labels=[1,-1])
       
        
        #test_metrics
        y_pred_test = best_model.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        f1_score_test = f1_score(y_test, y_pred_test, pos_label=1)
        confusion_matrix_test = confusion_matrix(y_test, y_pred_test, labels=[1,-1])
        print('accuracy_test: ',accuracy_test)
        print('f1_score_test: ',f1_score_test)
        
        metrics = {
            'Metric': ['Accuracy_val', 'F1_Score_val', 'Accuracy_test', 'F1_Score_test'],
            'Value': [accuracy_val, f1_score_val, accuracy_test, f1_score_test]
        }
        
        df_metrics = pd.DataFrame(metrics)
        df_metrics.to_csv('result/bagging/metrics_lof.csv', index=False)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix_val, annot=True, cmap='Blues', fmt='g', xticklabels=[1, -1], yticklabels=[1, -1])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix Validation Set')
        plt.savefig('result/bagging/confusion_matrix_validation.png', dpi=300, bbox_inches='tight')
        
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix_test, annot=True, cmap='Blues', fmt='g', xticklabels=[1, -1], yticklabels=[1, -1])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix Test Set')
        plt.savefig('result/bagging/confusion_matrix_test.png', dpi=300, bbox_inches='tight')
    
    
    def boosting(self, classifier, X_train, X_val, X_test, y_train, y_val, y_test):
        
        boosting_model = AdaBoostClassifier()

        param_grid = {
            'base_estimator': classifier,
            'n_estimators': list(range(1,4)),
            'learning_rate':  np.linspace(0.1, 1, 5),
            'random_state':42
        }
        
        f1_scorer = make_scorer(f1_score, average='binary')
        grid_search = GridSearchCV(estimator=boosting_model, param_grid=param_grid, scoring=f1_scorer, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred_val = best_model.predict(X_val)
        opt_params = {
                'Best_Parameter': ['Best_params', 'Best_F1_score_val'],
                'Value': [grid_search.best_params_, grid_search.best_score_]
            }
            
        df_best_params = pd.DataFrame(opt_params)
        df_best_params.to_csv('result/boosting/best_params_lof.csv', index=False)
        
        #validation_metrics
        accuracy_val = accuracy_score(y_val, y_pred_val)
        f1_score_val = f1_score(y_val, y_pred_val, pos_label=1)
        confusion_matrix_val = confusion_matrix(y_val, y_pred_val, labels=[1,-1])
       
        
        #test_metrics
        y_pred_test = best_model.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        f1_score_test = f1_score(y_test, y_pred_test, pos_label=1)
        confusion_matrix_test = confusion_matrix(y_test, y_pred_test, labels=[1,-1])
        print('accuracy_test: ',accuracy_test)
        print('f1_score_test: ',f1_score_test)
        
        metrics = {
            'Metric': ['Accuracy_val', 'F1_Score_val', 'Accuracy_test', 'F1_Score_test'],
            'Value': [accuracy_val, f1_score_val, accuracy_test, f1_score_test]
        }
        
        df_metrics = pd.DataFrame(metrics)
        df_metrics.to_csv('result/boosting/metrics_lof.csv', index=False)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix_val, annot=True, cmap='Blues', fmt='g', xticklabels=[1, -1], yticklabels=[1, -1])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix Validation Set')
        plt.savefig('result/boosting/confusion_matrix_validation.png', dpi=300, bbox_inches='tight')
        
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix_test, annot=True, cmap='Blues', fmt='g', xticklabels=[1, -1], yticklabels=[1, -1])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix Test Set')
        plt.savefig('result/boosting/confusion_matrix_test.png', dpi=300, bbox_inches='tight')
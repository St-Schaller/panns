import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import librosa
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels
import glob
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import PredefinedSplit
from hypopt import GridSearch
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import recall_score,f1_score, make_scorer
from sklearn.utils import resample



'''

def grid_search_with_val():
    df_train = pd.read_csv('Cold_Train_merged_with_labels_panns.csv')
    df_devel = pd.read_csv('Cold_Devel_merged_with_labels_panns.csv')
    df_test = pd.read_csv('Cold_Test_merged_with_labels_panns.csv')
    print(df_train)
    print(df_devel)
    print(df_test)

    X_train = df_train.iloc[:, 1:-1]
    print(X_train)
    y_train = df_train.iloc[:, -1:]
    print(y_train)
    X_devel = df_devel.iloc[:, 1:-1]
    print(X_devel)
    y_devel = df_devel.iloc[:, -1:]
    print(y_devel)
    X_test = df_test.iloc[:, 1:-1]
    print(X_test)
    y_test = df_test.iloc[:, -1:]
    print(y_test)

    param_grid = {
        'scaler__feature_range': [(-1, 1)],
        'dimensionality_reduction__n_components': [200],
        'classifier__C': [0.1]
        # 'scaler__feature_range': [(0, 1), (-1, 1)],
        # 'dimensionality_reduction__n_components': [8, 16, 32, 100, 200],
        # 'classifier__C': np.logspace(0, -8, num=9)
    }

    pipeline = Pipeline(steps=[('scaler', MinMaxScaler()),
                               ('dimensionality_reduction', PCA()),
                               ('classifier', LinearSVC(max_iter=10000))])
    # Grid-search all parameter combinations using a validation set.
    opt = GridSearch(model=pipeline, param_grid=param_grid)
    opt.fit(X_train, y_train.values.ravel(), X_devel, y_devel)
    print(f'Best Parameters: {opt.best_params}')
    print(f'Best Score on Devel: {opt.best_score}')
    print('Test Score for Optimized Parameters:', opt.score(X_test, y_test))
    preds = opt.predict(X_test)
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))
    print(confusion_matrix(y_test, preds, labels=["C", "NC"]))
    sns.heatmap(confusion_matrix(y_test, preds), annot=True)
    plt.show()
    
    
'''


def extract_data():
    data = []
    names = []
    device = 'cpu' # 'cuda' | 'cpu'

    colnames = ['file_name', 'labels']
    labels = pd.read_csv('/home/local/Dokumente/HeartApp/physionet_challenge/labels_combined.csv', names=colnames, header=None)
    print(labels)

    for audio_path in sorted(glob.iglob('/home/local/Dokumente/HeartApp/physionet_challenge/validation/*.wav')):
        (audio, _) = librosa.core.load(audio_path, sr=32000, mono=True)
        audio = audio[None, :]  # (batch_size, segment_samples)
        print(type(audio))
        print(audio.shape)
        at = AudioTagging(checkpoint_path=None, device=device)
        (clipwise_output, embedding) = at.inference(audio)
        """clipwise_output: (batch_size, classes_num), embedding: (batch_size, embedding_size)"""
        file = Path(audio_path).stem
        names.append(file)
        print(file)
        #print(embedding[0])
        data.append(embedding[0])

    data_np = pd.DataFrame(data)
    data_np.insert(0, 'file_name', names)
    print(data_np)
    df_merged = pd.merge(data_np, labels, how='inner', on='file_name')
    print(df_merged)
    df_merged.to_csv('/home/local/Dokumente/HeartApp/physionet_challenge/Physionet_allwav_validation_panns_CNN14.csv', index=False)



def train_model():
    df_train = pd.read_csv('Cold_Train_merged_with_labels_panns_wavegram.csv')
    df_devel = pd.read_csv('Cold_Devel_merged_with_labels_panns_wavegram.csv')
    df_test = pd.read_csv('Cold_Test_merged_with_labels_panns_wavegram.csv')
    df_train = pd.concat([df_train, df_devel], ignore_index=True)

    print(df_train)

    X_train = df_train.iloc[:, 1:-1]
    print(X_train)
    y_train = df_train.iloc[:, -1:]
    print(y_train)
    #X_devel = df_devel.iloc[:, 1:-1]
    #print(X_devel)
    X_test = df_test.iloc[:, 1:-1]
    print(X_test)
    #y_devel = df_devel.iloc[:, -1:]
    #print(y_devel)
    y_test = df_test.iloc[:, -1:]
    print(y_test)

    '''
        print(df_train['Snore'].value_counts())
        max = df_train['Snore'].value_counts().max()
        v_class = df_train[df_train.Snore == 'V']
        print(v_class)
        o_class = df_train[df_train.Snore == 'O']
        t_class = df_train[df_train.Snore == 'T']
        e_class = df_train[df_train.Snore == 'E']

        o_class = resample(o_class, replace=True, n_samples=max, random_state=120)
        t_class = resample(t_class, replace=True, n_samples=max, random_state=120)
        e_class = resample(e_class, replace=True, n_samples=max, random_state=120)

        df_train = pd.concat([v_class, o_class, t_class, e_class], ignore_index=True)
        print(df_train['Snore'].value_counts())
        print(df_train)
        '''
    df_train = df_train.rename(columns={'Cold (upper respiratory tract infection)': 'Cold'})
    print(df_train['Cold'].value_counts())
    max = df_train['Cold'].value_counts().max()
    nc_class = df_train[df_train.Cold == 'NC']
    print(nc_class)
    cold_class = df_train[df_train.Cold == 'C']

    cold_class = resample(cold_class, replace=True, n_samples=max, random_state=120)

    df_train = pd.concat([nc_class, cold_class], ignore_index=True)
    print(df_train['Cold'].value_counts())
    print(df_train)





    param_grid = {
        'scaler__feature_range': [(0, 1)],
        'dimensionality_reduction__n_components': [200],
        'classifier__C': [1.0]
    }

    pipeline = Pipeline(steps=[('scaler', MinMaxScaler()),
                               ('dimensionality_reduction', PCA()),
                               ('classifier', LinearSVC(max_iter=10000,class_weight='balanced'))])
    UAR = make_scorer(recall_score, average='macro')
    gs = GridSearchCV(pipeline, cv=2, scoring=UAR, refit=True, param_grid=param_grid, n_jobs=-1,
                      verbose=3)
    gs.fit(X_train, y_train.values.ravel())
    print(f'Best Train score: {gs.best_score_}')
    #preds_devel = gs.predict(X_devel)
    preds_test = gs.predict(X_test)
    #print(f"Devel Score: { recall_score(y_devel, preds_devel, average='macro')}")
    print(f"Test Score: {recall_score(y_test, preds_test, average='macro')}")
    #print(f"Best Test Score: {gs.best_estimator_.score(X_test, y_test)}")
    print(f'Best Parameters: {gs.best_params_}')

    print(classification_report(y_test, preds_test))
    #disp = plot_confusion_matrix(gs, X_test, y_test, display_labels=["E", "O", "T", "V"], cmap=plt.cm.Blues, normalize=None)
    disp = plot_confusion_matrix(gs, X_test, y_test, display_labels=["C", "NC"], cmap=plt.cm.Blues, normalize=None)
    disp.ax_.set_title('Cold PANNs upsampled')
    print(disp.confusion_matrix)
    plt.savefig('conf_matrix_cold_panns_wavegram_with_upsampled.png')
    plt.show()



def grid_search():
    df_train = pd.read_csv('/home/local/Dokumente/HeartApp/physionet_challenge/Physionet_allwav_train_panns_CNN14.csv')
    df_test = pd.read_csv('/home/local/Dokumente/HeartApp/physionet_challenge/Physionet_allwav_validation_panns_CNN14.csv')
    print(df_train)
    print(df_test)
    df_train = df_train[~df_train.file_name.isin(df_test['file_name'])]

    X_train = df_train.iloc[:, 1:-1]
    print(X_train)
    y_train = df_train.iloc[:, -1:]
    print(y_train)
    X_test = df_test.iloc[:, 1:-1]
    print(X_test)
    y_test = df_test.iloc[:, -1:]
    print(y_test)

    pipelines = [
        Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('classifier', LinearSVC(max_iter=10000,class_weight='balanced'))
            ]
        ),
        Pipeline(
            steps=[
                ('scaler', StandardScaler()),
                ('dimensionality_reduction', PCA()),
                ('classifier', LinearSVC(max_iter=10000,class_weight='balanced'))
            ]
        ),
        Pipeline(
            steps=[
                ('scaler', MinMaxScaler()),
                ('dimensionality_reduction', PCA()),
                ('classifier', LinearSVC(max_iter=10000,class_weight='balanced'))
            ]
        )
    ]

    parameter_grids = [
        {
            'classifier__C': [0.001, 0.1, 1.0]
        },
        {
            'dimensionality_reduction__n_components': [32, 100, 200],
            'classifier__C': [0.001, 0.1, 1.0]
        },
        {
            'scaler__feature_range': [(0, 1), (-1, 1)],
            'dimensionality_reduction__n_components': [32, 100, 200],
            'classifier__C': [0.001, 0.1, 1.0]
        }
    ]
    scores_best_estimators = []
    params_best_estimators = []
    best_estimators = []
    best_scores = []
    for i in range(len(pipelines)):
        gs = GridSearchCV(pipelines[i], parameter_grids[i], scoring='recall_macro', n_jobs=-1, cv=8, verbose=3)
        gs.fit(X_train, y_train.values.ravel())
        best_scores.append(gs.best_score_)
        scores_best_estimators.append(gs.best_estimator_.score(X_test, y_test))
        params_best_estimators.append(gs.best_params_)
        best_estimators.append(gs.best_estimator_)

    print(best_scores)
    print(scores_best_estimators)
    print(params_best_estimators)
    print(best_estimators)



if __name__ == '__main__':
    #extract_data()
    # grid_search_with_val()
    #add_test_labels()
    #train_model()
    grid_search()




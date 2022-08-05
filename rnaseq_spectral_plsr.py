#!/usr/bin/env python3
"""
Author : Emmanuel Gonzalez
Date   : 2022-03-31
Purpose: Partial least squares regression (PLSR) on gene expression (response variable) and spectral data (explanatory variable).
"""

import argparse
import os 
import sys
from sys import stdout
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.model_selection import permutation_test_score
from scipy.signal import savgol_filter
import matplotlib.collections as collections
import pickle
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import SelectFromModel
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold#, f_regression, mutual_info_regression, SelectFdr
from statistics import mean




# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='RNA-Seq & Spectral partial least squares regression (PLSR)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-c',
                        '--transcript_csv',
                        help='CSV for a single transcript containing all spectra.',
                        metavar='str',
                        type=str,
                        required=True)

    parser.add_argument('-pod',
                        '--plot_out_dir',
                        help='Plot/figure output directory.',
                        metavar='str',
                        type=str,
                        default='plots')

    parser.add_argument('-cod',
                        '--csv_out_dir',
                        help='A named integer argument',
                        metavar='str',
                        type=str,
                        default='correlation_scores')

    parser.add_argument('-o',
                        '--outdir',
                        help='Main output directory',
                        metavar='str',
                        type=str,
                        default='plsr_outputs')

    parser.add_argument('-m',
                        '--model_out_dir',
                        help='Model output directory',
                        metavar='str',
                        type=str,
                        default='models')

    parser.add_argument('-p',
                        '--permutation_out_dir',
                        help='Permutation output directory',
                        metavar='str',
                        type=str,
                        default='permutation')

    parser.add_argument('-t',
                        '--test_size',
                        help='Test size for train/test split.',
                        metavar='float',
                        type=float,
                        default=0.20)
    
    parser.add_argument('-maxw',
                        '--max_wavelength',
                        help='Maximum wavelength value.',
                        metavar='int',
                        type=int,
                        default=2500)

    parser.add_argument('-minw',
                        '--min_wavelength',
                        help='Minimum wavelength value.',
                        metavar='int',
                        type=int,
                        default=350)

    parser.add_argument('-vt',
                        '--variance_threshold',
                        help='Variance threshold.',
                        metavar='int',
                        type=int,
                        default=1)

    parser.add_argument('-rns',
                        '--random_number_seed',
                        help='Variance threshold.',
                        metavar='int',
                        type=int,
                        default=123)

    parser.add_argument('-oncfn',
                        '--onc_file_name',
                        help='Filename for optimal number of components output file.',
                        metavar='str',
                        type=str,
                        default='find_optimal_number_components')

    parser.add_argument('-oncmt',
                        '--onc_max_tests',
                        help='Maximum number of tests',
                        metavar='int',
                        type=int,
                        default=30)

    parser.add_argument('-np',
                        '--number_permutations',
                        help='Number of permutations to run',
                        metavar='int',
                        type=int,
                        default=1000)

    return parser.parse_args()


# --------------------------------------------------
def create_output_directories(transcript):

    args = get_args()

    #Define plot/figure output directory
    global plot_out_dir 
    plot_out_dir = os.path.join(args.outdir, args.plot_out_dir, transcript)
    if not os.path.isdir(plot_out_dir):
        try:
            os.makedirs(plot_out_dir)
        except OSError:
            pass
        
    # Define CSV output directory
    global csv_out_dir
    csv_out_dir = os.path.join(args.outdir, args.csv_out_dir, transcript)
    if not os.path.isdir(csv_out_dir):
        try:
            os.makedirs(csv_out_dir)
        except OSError:
            pass

    global model_out_dir
    model_out_dir = os.path.join(args.outdir, args.model_out_dir, transcript)
    if not os.path.isdir(model_out_dir):
        try:
            os.makedirs(model_out_dir)
        except OSError:
            pass
    
    # global permutation_out_dir
    # permutation_out_dir = os.path.join(args.outdir, args.permutation_out_dir, transcript)
    # if not os.path.isdir(permutation_out_dir):
    #     try:
    #         os.makedirs(permutation_out_dir)
    #     except OSError:
    #         pass


# --------------------------------------------------
def train_plsr(ncomp, X_train, y_train, X_test, y_test):

    args = get_args()
    # Run PLS with suggested number of components
    pls = PLSRegression(n_components=ncomp)
    pls.fit(X_train, y_train)
    
    # Calculate scores for train calibration and test
    score_train = pls.score(X_train, y_train)
    score_test = pls.score(X_test, y_test)

    # Calculate mean square error for calibration and cross validation
    mse_train = mean_squared_error(y_train, pls.predict(X_train))
    mse_test = mean_squared_error(y_test, pls.predict(X_test))

    # print('Train R2: %5.3f'  % score_train)
    # print('Train MSE: %5.3f' % mse_train)
    # print('Test R2: %5.3f'  % score_test)
    # print('Test MSE: %5.3f' % mse_test)

    return score_train, score_test, mse_train, mse_test, pls


# --------------------------------------------------
def save_plsr_model(filename, model):
    # Save to file in the current working directory
    
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


# --------------------------------------------------
def open_plsr_model(filename):
    # Load from file
    with open(filename, 'rb') as file:
        model = pickle.load(file)


    return model


# --------------------------------------------------
def scale_data(data):

    args = get_args()
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)

    return scaled_data


# --------------------------------------------------
def find_optimal_number_components(X_train, y_train, X_test, y_test, transcript):

    args = get_args()
    # Run PLSR
    result_dict = {}

    for i in range(1, args.onc_max_tests+1):
  
        score_train, score_test, mse_train, mse_test, model = train_plsr(ncomp=i, 
                                                                  X_train=X_train, 
                                                                  y_train=y_train,
                                                                  X_test=X_test, 
                                                                  y_test=y_test)
        
        mean_permutation_score, mean_permutation_mse, mean_permutation_rmse = run_permutation_test(X_test=X_test, y_test=y_test, model=model)

        result_dict[i] = {
            'number_of_components': int(i),

            'score_train_test_delta': abs(score_train - score_test),
            'score_train': score_train,
            'score_test': score_test,
            'score_test_mean_permutation': mean_permutation_score,

            'rmse_train_test_delta': abs(np.sqrt(mse_train) - np.sqrt(mse_test)),
            'rmse_train': np.sqrt(mse_train),
            'rmse_test': np.sqrt(mse_test),
            'rmse_test_mean_permutation': mean_permutation_rmse,

            'mse_train_test_delta': abs(mse_train - mse_test),
            'mse_train': mse_train, 
            'mse_test': mse_test,
            'mse_test_mean_permutation': mean_permutation_mse
        }

        save_plsr_model(filename=os.path.join(model_out_dir, '.'.join(['_'.join([transcript, str(i)]), "pkl"])), model=model)

    df = pd.DataFrame.from_dict(result_dict, orient='index').sort_values('rmse_train_test_delta')
    selected_components = int(df.iloc[0]['number_of_components'])  

    df['transcript'] = transcript
    df = df.set_index('number_of_components')
    df['selected'] = False
    df.at[selected_components, 'selected'] = True

    return df.reset_index(), selected_components


# --------------------------------------------------
def variance_threshold_variable_selection(data, y, threshold, transcript):

    args = get_args()
    selector = VarianceThreshold(threshold=threshold)
    selector.fit_transform(data)
    selected_data = data[data.columns[selector.get_support()]]
    
    selected_bands = pd.DataFrame()
    selected_bands['bands'] = selector.get_feature_names_out()

    selected_bands.to_csv(os.path.join(csv_out_dir, '.'.join(['_'.join([transcript, 'selected_bands']), 'csv'])), index=False)

    return selected_data


# --------------------------------------------------
def create_delta_figure(df, transcript, optimal_components):
    
    score = df[['number_of_components', 'score_train_test_delta', 'rmse_train_test_delta']]
    score = score.set_index('number_of_components').melt(ignore_index=False).reset_index()
    score = score.rename(columns={'variable': 'Metric'})

    remap_dict = {'score_train_test_delta': 'R$^2$',
                'rmse_train_test_delta': 'RMSE'}

    score['Metric'] = score['Metric'].map(remap_dict)

    sns.relplot(x='number_of_components', 
                y='value',  
                hue='Metric',
                style='Metric',
                markers=True, 
                kind='line', 
                data=score)
                
    plt.ylabel('|$\Delta$ train, test|')
    plt.xlabel('Number of PLSR components')
    plt.axvline(optimal_components, c='r')
    plt.savefig(os.path.join(plot_out_dir, '.'.join(['_'.join([transcript, 'delta']), 'png'])), dpi=1000, bbox_inches='tight', facecolor='w', edgecolor='w')


# --------------------------------------------------
def create_score_figure(df, transcript, optimal_components):
    
    score = df[['number_of_components', 'score_train', 'score_test']]
    score = score.set_index('number_of_components').melt(ignore_index=False).reset_index()
    score = score.rename(columns={'variable': 'Dataset'})

    remap_dict = {'score_train': 'Train',
                'score_test': 'Test'}

    score['Dataset'] = score['Dataset'].map(remap_dict)

    sns.relplot(x='number_of_components', 
                y='value', 
                hue='Dataset', 
                kind='line', 
                data=score)
    plt.axvline(optimal_components, c='r')
    plt.ylabel('R$^2$')
    plt.xlabel('Number of PLSR components')
    plt.axvline(optimal_components, c='r')
    plt.savefig(os.path.join(plot_out_dir, '.'.join(['_'.join([transcript, 'score']), 'png'])), dpi=1000, bbox_inches='tight', facecolor='w', edgecolor='w')


# --------------------------------------------------
def run_permutation_test(X_test, y_test, model):

    args = get_args()
    # Run permutation 
    cnt = 0
    permutation_score_list = []
    permutation_mse_list = []
    permutation_rmse_list = []

    for i in range(1, args.number_permutations+1):
        
        shuffled_X_test = shuffle(X_test)#, random_state=args.random_number_seed)
        # model = open_plsr_model(filename=os.path.join(model_out_dir, '.'.join(['_'.join([transcript, 'final']), "pkl"])))
        permutation_score_test = model.score(shuffled_X_test, y_test)
        permutation_mse_test = mean_squared_error(y_test, model.predict(shuffled_X_test))
        permutation_rmse_test = np.sqrt(permutation_mse_test)

        permutation_score_list.append(permutation_score_test)
        permutation_mse_list.append(permutation_mse_test)
        permutation_rmse_list.append(permutation_rmse_test)

    mean_permutation_score = mean(permutation_score_list)
    mean_permutation_mse = mean(permutation_mse_list)
    mean_permutation_rmse = mean(permutation_rmse_list)
    # print(f'[RESULT] Permutation test R2: {mean_permutation_score}')

    return mean_permutation_score, mean_permutation_mse, mean_permutation_rmse


# --------------------------------------------------
def get_derivative(X):

    X1 = savgol_filter(X, 11, polyorder = 2, deriv=1)
    X2 = savgol_filter(X, 13, polyorder = 2,deriv=2)

    return X1, X2


# --------------------------------------------------
def plsr_component_optimization(df, transcript, rng):

    args = get_args()
    
    print(f'[INFO] Running PLSR component optimization: {transcript}.')
    
    # Prepare explanatory/independent and response/dependent variables
    y = df[[transcript]]
    X = df[[str(i) for i in range(args.min_wavelength, args.max_wavelength+1)]]

    # Calculate derivatives, scale data, and apply variance threshold
    first_deriv, second_deriv = get_derivative(X)
    X = pd.DataFrame(first_deriv, columns = X.columns)
    # print(X)
    X = scale_data(X)
    X = variance_threshold_variable_selection(data=X, y=y, threshold=args.variance_threshold, transcript=transcript)
    print(f'[INFO] Variables selected: {len(X.columns)}')
    print('[INFO] Scaling data using StandardScaler.')

    # Created raw(train & test), selected(traing & test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng, test_size=args.test_size)

    # Find optimal number of PLSR components
    df, n_comp = find_optimal_number_components(X_train, y_train, X_test, y_test, transcript=transcript)
    print(f'[RESULT] Optimal number of components: {n_comp}')
    df.to_csv(os.path.join(csv_out_dir, '.'.join(['_'.join([transcript, args.onc_file_name]), 'csv'])), index=False)

    create_delta_figure(df=df, transcript=transcript, optimal_components=n_comp)
    create_score_figure(df=df, transcript=transcript, optimal_components=n_comp)
    
    # Run PLSR with the calculated optimal number of components
    final_score_train, final_score_test, final_mse_train, final_mse_test, model = train_plsr(n_comp, 
                                                                                      X_train=X_train, 
                                                                                      y_train=y_train, 
                                                                                      X_test=X_test, 
                                                                                      y_test=y_test)
    
    # Save the optimal PLSR model
    save_plsr_model(filename=os.path.join(model_out_dir, '.'.join(['_'.join([transcript, 'final']), "pkl"])), model=model)
    print(f'[RESULT] Train R2:{final_score_train}\n[RESULT] Test R2: {final_score_test}')

    


# --------------------------------------------------
def main():
    """Run PLSR here"""

    args = get_args()

    # Open CSV and get transcript name
    df = pd.read_csv(args.transcript_csv)
    transcript = '_'.join(os.path.basename(args.transcript_csv).split('_')[:2])

    # Create output directories
    create_output_directories(transcript=transcript)

    # Run PLSR and variable selection
    plsr_component_optimization(df=df, transcript=transcript, rng=args.random_number_seed)


# --------------------------------------------------
if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Author : Emmanuel Gonzalez
Date   : 2022-03-31
Purpose: Partial least squares regression (PLSR) on gene expression (response variable) and spectral data (explanatory variable).
"""

import argparse
# from math import perm
import os 
import sys
from sys import stdout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold, permutation_test_score
import matplotlib.collections as collections
import pickle
import multiprocessing
import itertools
import warnings
warnings.filterwarnings("ignore")


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

    parser.add_argument('-t',
                        '--test_size',
                        help='Test size for train/test split.',
                        metavar='float',
                        type=float,
                        default=0.25)

    return parser.parse_args()


# --------------------------------------------------
def pls_variable_selection(X, y, max_comp, transcript):
    
    # Define MSE array to be populated
    mse = np.zeros((max_comp,X.shape[1]))
 
    # Loop over the number of PLS components
    for i in range(max_comp):
        
        # Regression with specified number of components, using full spectrum
        pls1 = PLSRegression(n_components=i+1)
        pls1.fit(X, y)
        
        # Indices of sort spectra according to ascending absolute value of PLS coefficients
        sorted_ind = np.argsort(np.abs(pls1.coef_[:,0]))
 
        # Sort spectra accordingly 
        Xc = X[:,sorted_ind]
 
        # Discard one wavelength at a time of the sorted spectra, regress, and calculate the MSE cross-validation
        for j in range(Xc.shape[1]-(i+1)):
 
            pls2 = PLSRegression(n_components=i+1)
            pls2.fit(Xc[:, j:], y)
            
            y_cv = cross_val_predict(pls2, Xc[:, j:], y, cv=5)
 
            mse[i,j] = mean_squared_error(y, y_cv)
    
        comp = 100*(i+1)/(max_comp)
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")
 
    # Calculate and print the position of minimum in MSE
    global mseminy
    mseminx,mseminy = np.where(mse==np.min(mse[np.nonzero(mse)]))
 
    print("Optimized number of PLS components: ", mseminx[0]+1)
    print("Wavelengths to be discarded ",mseminy[0])
    print('Optimized MSEP ', mse[mseminx,mseminy][0])
    stdout.write("\n")
    # plt.imshow(mse, interpolation=None)
    # plt.savefig(os.path.join(plot_out_dir, f'{transcript}_mse_plot.png'), transparent=True)
    # plt.show()
 
    # Calculate PLS with optimal components and export values
    pls = PLSRegression(n_components=mseminx[0]+1)
    pls.fit(X, y)
        
    sorted_ind = np.argsort(np.abs(pls.coef_[:,0]))
 
    Xc = X[:,sorted_ind]
 
    return(Xc[:,mseminy[0]:],mseminx[0]+1,mseminy[0], sorted_ind)


def func(args):
    i, x_to_permute, n_comp = args
    np.random.seed(i + x_to_permute*1000)

    # permute x_train for feature x_to_permute
    perm_idx = np.random.choice(len(X_train), len(X_train), False)
    x_train_p = X_train.copy()
    x_train_p[:, x_to_permute] = x_train_p[perm_idx, x_to_permute]

    # bts_est = ExtraTreesClassifier()
    # bts_est.fit(x_train_p, y_train.ravel())
    pls = PLSRegression(n_components=n_comp)
    pls.fit(x_train_p, y_train)

    # permute x_val for feature x_to_permute
    perm_idx = np.random.choice(len(X_test), len(X_test), False)
    x_val_p = X_test.copy()
    x_val_p[:, x_to_permute] = x_val_p[perm_idx, x_to_permute]

    score = pls.score(x_val_p, y_test)

    return score


def get_scores_of_permuted_features(X_train, n_comp):
    scores_permuted = []
    cpu_count = os.cpu_count()

    for x_to_permute in range(X_train.shape[1]):
        score = multiprocessing.Pool(cpu_count).map(func, itertools.product(range(99), [x_to_permute], [n_comp] * 50))
        scores_permuted.append(score)
    return np.array(scores_permuted).T


# --------------------------------------------------
def simple_pls_cv(X, y, n_comp, transcript, rng=123, trained_model=None):#, permutation=False):

    args = get_args()
    if trained_model:
        print('Trained model')
        filename = os.path.join(model_out_dir, '_'.join([transcript, 'model.sav']))
        pls = pickle.load(open(filename, 'rb'))
        result = pls.score(X, y)
        print(result)

    else:
        # Split data into train & test datasets
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng, test_size=args.test_size)
        
        # Run PLS with suggested number of components
        pls = PLSRegression(n_components=n_comp)
        pls.fit(X_train, y_train)
        
        # Calculate scores for train calibration and test
        score_train = pls.score(X_train, y_train)
        score_test = pls.score(X_test, y_test)
    
        # Calculate mean square error for calibration and cross validation
        mse_train = mean_squared_error(y_train, pls.predict(X_train))
        mse_test = mean_squared_error(y_test, pls.predict(X_test))

        # if permutation:
        #     print('Running permutation test')
        #     scores_permuted = get_scores_of_permuted_features(X_train, n_comp)
        #     print(scores_permuted)
    
        print('Train R2: %5.3f'  % score_train)
        print('Train MSE: %5.3f' % mse_train)
        print('Test R2: %5.3f'  % score_test)
        print('Test MSE: %5.3f' % mse_test)
        res_dict = {}
        
        res_dict[transcript] = {
            'score_train': score_train, 
            'score_test': score_test,
            'mse_train': mse_train, 
            'mse_test': mse_test,
            'number_components': n_comp
        }
        
        res_df = pd.DataFrame.from_dict(res_dict, orient='index')
        res_df.index.name = 'transcript'
    
        # Plot regression 
        z = np.polyfit(y_test, pls.predict(X_test), 1)
        with plt.style.context(('ggplot')):
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.scatter(pls.predict(X_test), y_test, c='red', edgecolors='k')
            ax.plot(z[1]+z[0]*y, y, c='blue', linewidth=1)
            ax.plot(y, y, color='green', linewidth=1, linestyle='dashed')
            plt.title('$R^{2}$ (CV): '+str(score_test))
            plt.xlabel('Predicted')
            plt.ylabel('Measured')
            plt.savefig(os.path.join(plot_out_dir, f'{transcript}_simple_pls_{n_comp}.png'), transparent=True)
            # plt.show()

        return res_df, pls


# --------------------------------------------------
def run_variable_selection(df, transcript):

    # Collect response (RNAseq TPM) and explanatory variables (spectra) for a single transcript
    print(f'Processing transcript: {transcript}')
    y = df[transcript].values
    X = df[[str(i) for i in range(350, 2501)]]
    wl = np.arange(350, 2501, 1)
    
    
    # Calculate the first and second derivatives
    X1 = savgol_filter(X, 11, polyorder = 2, deriv=1)
    X2 = savgol_filter(X, 13, polyorder = 2,deriv=2)
    
    # Standardize data
#     X1 = StandardScaler().fit_transform(X1)
#     X2 = StandardScaler().fit_transform(X2)

    # Define the PLS regression object & fit data
    pls = PLSRegression(n_components=8)
    pls.fit(X1, y)

    # Plot spectra
    plt.figure(figsize=(8,9))
    with plt.style.context(('ggplot')):
        ax1 = plt.subplot(211)
        plt.plot(wl, X1.T)
        plt.ylabel('First derivative absorbance spectra')

        ax2 = plt.subplot(212, sharex=ax1)
        plt.plot(wl, np.abs(pls.coef_[:,0]))
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Absolute value of PLS coefficients')
        plt.savefig(os.path.join(plot_out_dir, f'{transcript}_first_derivative_absolute_value_pls_coeff.png'), transparent=True)

    # # Get the list of indices that sorts the PLS coefficients in ascending order of the absolute value
    # sorted_ind = np.argsort(np.abs(pls.coef_[:,0]))

    # # Sort spectra according to ascending absolute value of PLS coefficients
    # Xc = X1[:,sorted_ind]

    # Simple PLSR 
    simple_pls_cv(X1, y, 8, transcript=transcript)

    # Variable selection PLSR
    global ncomp
    global wav
    global sorted_ind

    opt_Xc, ncomp, wav, sorted_ind = pls_variable_selection(X1, y, 15, transcript=transcript)
    res_df, pls = simple_pls_cv(opt_Xc, y, ncomp, transcript=transcript)#, permutation=False)
    out_file = os.path.join(csv_out_dir, '_'.join([transcript, 'correlation_score.csv']))
    res_df.to_csv(out_file)

    # Plot spectra with superimpose selected bands
    ix = np.in1d(wl.ravel(), wl[sorted_ind][:wav])
    fig, ax = plt.subplots(figsize=(8,9))
    with plt.style.context(('ggplot')):
        ax.plot(wl, X1.T)
        plt.ylabel('First derivative absorbance spectra')
        plt.xlabel('Wavelength (nm)')
        
    collection = collections.BrokenBarHCollection.span_where(
        wl, ymin=-1, ymax=1, where=ix == True, facecolor='red', alpha=0.3)
    ax.add_collection(collection)
    plt.savefig(os.path.join(plot_out_dir, f'{transcript}_variable_selection.png'), transparent=True)
    
    # Save wavelength (used & removed)
    used_wavelengths = pd.DataFrame([wl, ix]).T.rename(columns={0: 'wavelength', 1: 'removed'})
    out_file = os.path.join(csv_out_dir, '_'.join([transcript, 'selected_wavelengths.csv']))
    used_wavelengths.to_csv(out_file, index=False)
    
    # Save PLSR model 
    filename = os.path.join(model_out_dir, '_'.join([transcript, 'model.sav']))
    pickle.dump(pls, open(filename, 'wb'))

    print(f'Done processing transcript: {transcript}')
    print('#----------------------------------------------------------------------------')


# --------------------------------------------------
def run_variable_selection_permutation(df, transcript):

    # Collect response (RNAseq TPM) and explanatory variables (spectra) for a single transcript
    print(f'Processing transcript: {transcript}')
    y = df[transcript].values
    print(y)
    X = df[[str(i) for i in range(350, 2501)]]
    rng = np.random.default_rng()
    X = rng.permutation(X, axis=1)
    # X = rng.permutation(X, axis=1)
    # X = np.random.permutation(X)
    # X = np.random.random((X.shape[0], X.shape[1]))
    print(f'Randomized data shape: {X.shape}')
    wl = np.arange(350, 2501, 1)
    
    
    # Calculate the first and second derivatives
    X1 = savgol_filter(X, 11, polyorder = 2, deriv=1)
    X2 = savgol_filter(X, 13, polyorder = 2,deriv=2)

    # Define the PLS regression object & fit data
    pls = PLSRegression(n_components=8)
    pls.fit(X1, y)

    # Plot spectra
    plt.figure(figsize=(8,9))
    with plt.style.context(('ggplot')):
        ax1 = plt.subplot(211)
        plt.plot(wl, X1.T)
        plt.ylabel('First derivative absorbance spectra')

        ax2 = plt.subplot(212, sharex=ax1)
        plt.plot(wl, np.abs(pls.coef_[:,0]))
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Absolute value of PLS coefficients')
        plt.savefig(os.path.join(plot_out_dir, f'{transcript}_first_derivative_absolute_value_pls_coeff_permutation.png'), transparent=True)
        # plt.show()

    # Simple PLSR 
    # simple_pls_cv(X1, y, 8, transcript=transcript)

    # Variable selection PLSR
    # opt_Xc, ncomp, wav, sorted_ind = pls_variable_selection(X1, y, 15, transcript=transcript)
        
    Xc = X1[:,sorted_ind]
    X = Xc[:,mseminy[0]:]

    filename = os.path.join(model_out_dir, '_'.join([transcript, 'model.sav']))
    res_df = simple_pls_cv(X, y, ncomp, transcript=transcript, trained_model=filename)#, permutation=False)
    out_file = os.path.join(csv_out_dir, '_'.join([transcript, 'correlation_score_permutation.csv']))
    res_df.to_csv(out_file)

    # Plot spectra with superimpose selected bands
    ix = np.in1d(wl.ravel(), wl[sorted_ind][:wav])
    fig, ax = plt.subplots(figsize=(8,9))
    with plt.style.context(('ggplot')):
        ax.plot(wl, X1.T)
        plt.ylabel('First derivative absorbance spectra')
        plt.xlabel('Wavelength (nm)')
        
    collection = collections.BrokenBarHCollection.span_where(
        wl, ymin=-1, ymax=1, where=ix == True, facecolor='red', alpha=0.3)
    ax.add_collection(collection)
    plt.savefig(os.path.join(plot_out_dir, f'{transcript}_variable_selection_permutation.png'), transparent=True)
    # plt.show()
    
    # Save wavelength (used & removed)
    # used_wavelengths = pd.DataFrame([wl, ix]).T.rename(columns={0: 'wavelength', 1: 'removed'})
    # out_file = os.path.join(csv_out_dir, '_'.join([transcript, 'selected_wavelengths.csv']))
    # used_wavelengths.to_csv(out_file, index=False)
    print(f'Done processing transcript: {transcript}')
    print('#----------------------------------------------------------------------------')


# --------------------------------------------------
def main():
    """Run PLSR here"""

    args = get_args()

    # Open CSV and get transcript name
    df = pd.read_csv(args.transcript_csv)
    transcript = '_'.join(os.path.basename(args.transcript_csv).split('_')[:2])

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
    
    # Run PLSR and variable selection
    run_variable_selection(df=df, transcript=transcript)
    run_variable_selection_permutation(df=df, transcript=transcript)


# --------------------------------------------------
if __name__ == '__main__':
    main()

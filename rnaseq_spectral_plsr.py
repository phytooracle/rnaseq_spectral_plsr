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
import matplotlib.pyplot as plt
import multiprocessing
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.collections as collections
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


# --------------------------------------------------
def simple_pls_cv(X, y, n_comp, transcript, rng=123):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)
    
    # Run PLS with suggested number of components
    pls = PLSRegression(n_components=n_comp)
    pls.fit(X_train, y_train)
#     y_c = pls.predict(X)
 
    # Cross-validation
#     y_cv = cross_val_predict(pls, X, y, cv=10)    
 
    # Calculate scores for calibration and cross-validation
#     score_c = r2_score(y, y_c)
#     score_cv = r2_score(y, y_cv)
    score_c = pls.score(X_train, y_train)
    score_cv = pls.score(X_test, y_test)
 
    # Calculate mean square error for calibration and cross validation
    mse_c = mean_squared_error(y_train, pls.predict(X_train))
    mse_cv = mean_squared_error(y_test, pls.predict(X_test))
 
    print('R2 calib: %5.3f'  % score_c)
    print('R2 CV: %5.3f'  % score_cv)
    print('MSE calib: %5.3f' % mse_c)
    print('MSE CV: %5.3f' % mse_cv)
    res_dict = {}
    
    res_dict[transcript] = {
        'calibration_score': score_c, 
        'cross_validation_score': score_cv,
        'calibration_mse': mse_c, 
        'cross_validation_mse': mse_cv,
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
        plt.title('$R^{2}$ (CV): '+str(score_cv))
        plt.xlabel('Predicted')
        plt.ylabel('Measured')
        plt.savefig(os.path.join(plot_out_dir, f'{transcript}_simple_pls_{n_comp}.png'), transparent=True)
        # plt.show()

    return res_df


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
        # plt.show()

    # Get the list of indices that sorts the PLS coefficients in ascending order of the absolute value
    sorted_ind = np.argsort(np.abs(pls.coef_[:,0]))

    # Sort spectra according to ascending absolute value of PLS coefficients
    Xc = X1[:,sorted_ind]

    # Simple PLSR 
    simple_pls_cv(X1, y, 8, transcript=transcript)

    # Variable selection PLSR
    opt_Xc, ncomp, wav, sorted_ind = pls_variable_selection(X1, y, 15, transcript=transcript)
    print(opt_Xc)
    print(sorted_ind)
    res_df = simple_pls_cv(opt_Xc, y, ncomp, transcript=transcript)
    out_file = os.path.join(csv_out_dir, '_'.join([transcript, 'correlation_score.csv']))
    res_df.to_csv(out_file)


    # Visualize discarded bands
    ix = np.in1d(wl.ravel(), wl[sorted_ind][:wav])
    print(ix)
    
    # Plot spectra with superimpose selected bands
    fig, ax = plt.subplots(figsize=(8,9))
    with plt.style.context(('ggplot')):
        ax.plot(wl, X1.T)
        plt.ylabel('First derivative absorbance spectra')
        plt.xlabel('Wavelength (nm)')
        
    collection = collections.BrokenBarHCollection.span_where(
        wl, ymin=-1, ymax=1, where=ix == True, facecolor='red', alpha=0.3)
    ax.add_collection(collection)
    plt.savefig(os.path.join(plot_out_dir, f'{transcript}_variable_selection.png'), transparent=True)
    # plt.show()
    
    used_wavelengths = pd.DataFrame([wl, ix]).T.rename(columns={0: 'wavelength', 1: 'removed'})
    out_file = os.path.join(csv_out_dir, '_'.join([transcript, 'selected_wavelengths.csv']))
    used_wavelengths.to_csv(out_file, index=False)
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
    
    # Run PLSR and variable selection
    run_variable_selection(df=df, transcript=transcript)


# --------------------------------------------------
if __name__ == '__main__':
    main()

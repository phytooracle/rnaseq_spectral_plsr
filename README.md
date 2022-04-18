# <p align="center">RNA-Seq & Hyperspectral PLSR</p>
This script (i) splits combined RNA-Seq & hyperspectral data for each transcript into train and test datasets, (ii) runs variable selection to find the lowest MSE, and trained/calibrated using gene expression as response variable and spectral data of selected bands as explanatory variable.

## Arguments
* -c, --transcript_csv
    * CSV for a single transcript containing all spectra

* -pod, --plot_out_dir
    * Plot/figure output directory

* -cod, --csv_out_dir
    * CSV output directory (calibration, test, and permutation scores)

* -o, --outdir
    * Main/root output directory

* -m, --model_out_dir
    * PLSR model sav file output directory

* -p, --permutation_out_dir
    * Permutation numpy array output directory

* -t, --test_size
    * Test size for train/test split (default=0.25)

# Running the script
To run the scrip, you have to specify at least the CSV:

```
./rnaseq_spectral_plsr.py -c ./rnaseq_spectra/Gh_A01G026200_tpm_spectra.csv
```
# licensePlateDatasetGenerator
Python package for generating a dataset composed of random french license plates. This dataset is meant for Deep Learning algorithm training for OCR and text recognition purposes.

The name of images are "LicensePlateNumber_NumberOfDepartement.png".

Font utilized for french LPs is FE-Schrift (current font utilized in European LPs).

## Usage
Run from command line.

The following arguments are accepted:

```
-n, --number_of_plates   Number of license plates to generate [default: 100]
-g, --gray               Choose if grayscale is used or not [default: False]
```

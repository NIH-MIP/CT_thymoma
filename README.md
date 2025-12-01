# Introduction
This software segments thymic epithelial tumors (TETs) in targetted regions of lungs (described later) and produces image renderings of the lung cell grid (described later), volume and RECIST measurements of the segmented TETs, and a post-processed CT image for some kind of 3D surface rendering in some CareStream viewers.

This repository hosts scripts to run inference thymoma segmentation on CT scans and the methodology is documented in the following publication:

`Choradia, Nirmal, Nathan Lay, Alex Chen, James Latanski, Meredith McAdams, Shannon Swift, Christine Feierabend et al. "Physician-guided deep learning model for assessing thymic epithelial tumor volume." Journal of Medical Imaging 12, no. 4 (2025): 046501-046501.`

https://doi.org/10.1117/1.JMI.12.4.046501

If you use any of our code, model weights or data for your research, please cite our paper.

# Models and Data
Model weights and training data not currently publicly available. But you may reach out to the corresponding author Chen Zhao at chen.zhao@nih.gov to request model weights and training data.

# Dependencies
You need the following python packages and tools:
* SimpleITK
* numpy
* opencv-python
* [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
* [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)

The first three can be installed via `pip`

`pip install SimpleITK opencv-python numpy`

# Usage
This package was primarily designed to run without interaction on Blackford or similar systems (e.g. cron?).

## nnU-Net TET Segmenter
The TET segmentation system is based on nnU-Net and requires you to set the following environment variables to the model weights
* `nnUNet_raw`
* `nnUNet_preprocessed`
* `nnUNet_results`

For the purpose of inference, most of these folders will be small with `nnUNet_results` holding the actual nnU-Net model weights.

## Cell Files
To achieve non-interactive execution, a CSV cell file and/or patient lung regions must first be created/defined. This is done is done two steps:
1) Run the TET segmenter with patient to produce a standalone cell grid.
   This can be done as follows:

   `python nnUNetCellInference.py --map-file '/path/to/cells.csv' --output-dir './Results' DICOMImageFolder`

   **NOTE**: If you don't have a cells file, you can create an empty text file to start with.
3) Identify the lung regions with TETs in the output cell image (series description "Tumor grid", series number 5001 by default) and input them into the cell file with following format

```csv
patient_id,cells
MRN1,comma-delimited regions
MRN2,"A2Y,A2Z,A3Y,A3Z"
...
```

Where, for example, `A2Y`, corresponds to a specific cuboid region in the cellular grid.

Finally, re-running the same image will produce two more DICOM series "Rendered volumetric measurements" (series number 5002 by default) and "HU tumor rendering" (series number 5003 by default). The former will show overlaid colorized TET segmentations with volumetric measurements and automatically-computed major/minor RECIST axes.

## Nifti
You can also run everything with Nifti files, but you must manually provide the patient ID to `nnUNetCellInference.py` with the `--patient-id` flag.


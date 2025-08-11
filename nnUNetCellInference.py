# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# August 2025
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

import argparse
import os
import csv
import SimpleITK as sitk
import numpy as np
import cv2
import glob
import functools
import operator
import shutil
#from render_results import RenderMeasurements, LoadDicomImage, SaveDicomImage, DrawForResearchUseOnly, _RenderPhantomLegendHU
from render_results import RenderMeasurements, LoadDicomImage, SaveDicomImage, DrawForResearchUseOnly, RenderSurfaceHU
import time
from contextlib import contextmanager

_rows = 5
_cols = 5
_levels = 5

# See here:
# https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time
#
@contextmanager
def catchtime(prefix="Elapsed time:"):
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
    print(f"{prefix} {time.perf_counter() - start:0.3f} seconds")

def binary_dilate(npMask, radius, foreground_value=1, background_value=0):
    dilateFilter = sitk.BinaryDilateImageFilter()

    dilateFilter.SetForegroundValue(foreground_value)
    dilateFilter.SetBackgroundValue(background_value)
    dilateFilter.SetKernelType(sitk.sitkBall)
    #dilateFilter.SetKernelType(sitk.sitkBox)
    dilateFilter.SetKernelRadius(radius)

    with catchtime("Binary dilate elapsed time:") as t:
        dilateMask = dilateFilter.Execute(sitk.GetImageFromArray(npMask))

    return sitk.GetArrayFromImage(dilateMask)

def safe_eval(list_str):
    list_str=list_str.replace("[","")
    list_str=list_str.replace("]","")
    list_str=list_str.replace("'","")
    list_str=list_str.replace(" ","")
    return list_str.split(",")

def load_cell_map(path):
    if not os.path.exists(path):
        return dict()

    with open(path, mode="rt", newline="") as f:
        reader = csv.DictReader(f)
        cell_map = { row["patient_id"]: safe_eval(row["cells"]) for row in reader }

    return cell_map

def save_cell_map(cell_map, path):
    rows = [ { "patient_id": patient_id, "cells": ",".join(sorted(list(cells))) } for patient_id, cells in cell_map.items() ]

    with open(path, mode="wt", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def load_image(image_path, dtype=sitk.sitkUnknown):
    if os.path.isdir(image_path):
        print(f"Info: Loading DICOM '{image_path}' ...", flush=True)
        img = LoadDicomImage(image_path, dtype=dtype)
        assert img is not None, f"LoadDicomImage() returned None for '{image_path}'"
        return img

    print(f"Info: Loading '{image_path}' ...", flush=True)
    return sitk.ReadImage(image_path, dtype)

def save_image(image, image_path, compress=True):
    if len(os.path.splitext(image_path)[1]) > 0:
        print(f"Info: Saving '{image_path}' ...", flush=True)
        sitk.WriteImage(image, image_path, useCompression=compress)
        return

    assert image.HasMetaDataKey("0020|000e")

    print(f"Info: Saving DICOM '{image_path}' ...", flush=True)
    SaveDicomImage(image, image_path, compress=compress)

def get_meta_data(image_path):
    reader = sitk.ImageFileReader()

    if os.path.isdir(image_path):
        reader.SetImageIO("GDCMImageIO")

        for dicom_file in glob.glob(os.path.join(image_path, "*")):
            if not os.path.isfile(dicom_file):
                continue

            try:
                reader.SetFileName(dicom_file)
                reader.ReadImageInformation()
                if reader.HasMetaDataKey("0020|000e"):
                    return reader
            except:
                continue
                
    else:
        try:
            reader.SetFileName(image_path)
            reader.ReadImageInformation()
            if reader.HasMetaDataKey("0020|000e"):
                return reader
        except:
            pass

    return None

def cell_neighbors(cell, connectivity=26):
    global _rows, _cols, _levels

    rows = _rows
    cols = _cols
    levels = _levels

    assert len(cell) == 3 and cell[0].lower() in "abcde" and cell[1] in "12345" and cell[2].lower() in "vwxyz"

    r0 = ord(cell[0].lower()) - ord('a')
    c0 = int(cell[1]) - 1
    l0 = ord(cell[2].lower()) - ord('v')

    if connectivity == 26:
        for dl in range(-1,2):
            l = l0 + dl
            if l < 0 or l >= levels:
                continue

            level_label = chr(ord('V') + l)

            for dr in range(-1,2):
                r = r0 + dr

                if r < 0 or r >= rows:
                    continue

                row_label = chr(ord('A') + r)

                for dc in range(-1,2):
                    if dl == 0 and dr == 0 and dc == 0:
                        continue

                    c = c0 + dc
                    if c < 0 or c >= cols:
                        continue

                    col_label = str(c+1)

                    yield row_label + col_label + level_label

    elif connectivity == 6:
        for dl in (-1,1):
            l = l0 + dl
            if l < 0 or l >= levels:
                continue
            
            level_label = chr(ord('V') + l)
            row_label = chr(ord('A') + r0)
            col_label = str(c0+1)

            yield row_label + col_label + level_label

        for dr in (-1,1):
            r = r0 + dr
            if r < 0 or r >= rows:
                continue

            level_label = chr(ord('V') + l0)
            row_label = chr(ord('A') + r)
            col_label = str(c0+1)

            yield row_label + col_label + level_label

        for dc in (-1,1):
            c = c0 + dc
            if c < 0 or c >= cols:
                continue

            level_label = chr(ord('V') + l0)
            row_label = chr(ord('A') + r0)
            col_label = str(c+1)

            yield row_label + col_label + level_label

    else:
        raise RuntimeError(f"No such connectivity: {connectivity}")

def cell_connected_components(cells):
    cells = set((cell.upper() for cell in cells))
    visited = set()

    def cell_seed():
        return next(iter((cell for cell in cells if cell not in visited)))

    cell_ccs = []

    while True:
        try:
            cell = cell_seed()
        except StopIteration:
            break

        cell_cc = [ cell ]
        visited.add(cell)

        i = 0
        while i < len(cell_cc):
            cell0 = cell_cc[i]
            i += 1

            for cell in cell_neighbors(cell0):
                if cell in cells and cell not in visited:
                    cell_cc.append(cell)
                    visited.add(cell)

        cell_ccs.append(cell_cc)        

    return cell_ccs

def resample(img, ref_img=None, new_spacing=None, new_size=None, interp="linear"):
    interp_dict = { 
        "linear": sitk.sitkLinear,
        "nearest": sitk.sitkNearestNeighbor,
        "bspline": sitk.sitkBSpline,
    }

    assert interp in interp_dict, f"Unknown interpolator '{interp}'. The following are supported: {interp_dict.keys()}"
    assert (ref_img is not None) + (new_spacing is not None) + (new_size is not None) == 1, "Only one of 'ref_img', 'new_spacing' or 'new_size' can be specified."

    interp = interp_dict[interp]

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interp)

    spacing = img.GetSpacing()
    size = img.GetSize()

    if ref_img is not None:
        resampler.SetReferenceImage(ref_img)
    elif new_spacing is not None:
        new_spacing = [ nsp if nsp > 0.0 else sp for sp, nsp in zip(spacing, new_spacing) ]
        new_size = [ round(sz*sp/nsp) for sp, sz, nsp in zip(spacing, size, new_spacing) ]

        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetOutputDirection(img.GetDirection())
    elif new_size is not None:
        new_size = [ nsz if nsz > 0 else sz for sz, nsz in zip(size, new_size) ]
        new_spacing = [ sz*sp/nsz for sp, sz, nsz in zip(spacing, size, new_size) ]

        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetOutputDirection(img.GetDirection())

    new_img = resampler.Execute(img)

    for key in img.GetMetaDataKeys():
        new_img.SetMetaData(key, img.GetMetaData(key))

    return new_img

def segment_lungs_and_others(image_path, output_path="lungs.nii.gz", device="gpu"):
    cmd=f"TotalSegmentator -i '{image_path}' -o '{output_path}' -d '{device}' --ml --roi_subset lung_upper_lobe_left lung_lower_lobe_left lung_upper_lobe_right lung_middle_lobe_right lung_lower_lobe_right"
    # Maybe esophagus?
    #cmd=f"TotalSegmentator -i '{image_path}' -o '{output_path}' -d '{device}' --ml --roi_subset lung_upper_lobe_left lung_lower_lobe_left lung_upper_lobe_right lung_middle_lobe_right lung_lower_lobe_right esophagus trachea heart aorta pulmonary_vein brachiocephalic_trunk superior_vena_cava"
    print(cmd)

    with catchtime("TotalSegmentator elapsed time:") as t:
        assert os.system(cmd) == 0, "Lung segmentation failed."

#def run_nnunet(input_dir, output_dir, device="cuda", dataset_name="Dataset300_TGMBCropsLarger", configuration="3d_fullres"):
#def run_nnunet(input_dir, output_dir, device="cuda", dataset_name="300", configuration="3d_fullres"):
def run_nnunet(input_dir, output_dir, device="cuda", dataset_name="Dataset307_TGMBCellCrops_box", configuration="3d_fullres"):
    cmd=f"nnUNetv2_predict -i '{input_dir}' -o '{output_dir}' -d '{dataset_name}' -c '{configuration}' -device '{device}'"
    print(cmd)

    with catchtime("nnU-Net elapsed time:") as t:
        assert os.system(cmd) == 0, "nnUNet failed"

def strip_ext(path):
    if path.lower().endswith(".nii.gz"):
        return path[:-7]

    return os.path.splitext(path)[0]

def get_lung_corners(np_lung_mask):
    p_begin, p_end = 0.0, 100.0 

    #lung_mask_indices = np.argwhere(np_lung_mask > 0)
    #lung_mask_indices = np.argwhere(np.logical_and(np_lung_mask > 0, np_lung_mask < 15))
    lung_mask_indices = np.argwhere(np.logical_and(np_lung_mask >= 10, np_lung_mask < 15))
    lower, upper = np.percentile(lung_mask_indices, (p_begin, p_end), axis=0)
    lower, upper = lower.astype(int), upper.astype(int)
    #size = upper - lower

    return lower, upper

def suppress_outside_of_lungs(mask, lung_mask):
    dilate_mm=5.0
    print(f"Info: Suppressing segmentations outside of lung mask ({dilate_mm} mm).")

    spacing = np.array(mask.GetSpacing())
    radius = [ int(x) for x in np.ceil(dilate_mm/spacing) ]

    print(f"Info: Mask spacing: {tuple(spacing)}")

    np_mask = sitk.GetArrayFromImage(mask)
    np_lung_mask = sitk.GetArrayViewFromImage(lung_mask)
    #np_heart_mask = (np_lung_mask >= 51)
    np_other_mask = (np_lung_mask >= 15)
    np_lung_mask = np.logical_and(np_lung_mask >= 10, np_lung_mask < 15)

    print(f"Info: Dilating lung mask by radius (x,y,z) = {radius} ...")
    np_lung_mask = binary_dilate(np_lung_mask.astype(np.uint8), radius=radius).astype(bool)

    np_lung_mask = np.logical_or(np_lung_mask, np_other_mask)

    with catchtime("Mask suppression elapsed time:") as t:
        # Do not dilate into heart
        #np_lung_mask[np_other_mask] = 0

        np_mask[np_lung_mask == 0] = 0

    new_mask = sitk.GetImageFromArray(np_mask)
    new_mask.CopyInformation(mask)

    return new_mask

def combine_crop_masks(ref_img, input_dir, mask=None):
    resampler = sitk.ResampleImageFilter()

    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetReferenceImage(ref_img)
    resampler.SetOutputPixelType(sitk.sitkInt16)

    if mask is None:
        new_mask = np.zeros(list(ref_img.GetSize())[::-1], dtype=np.int16)
    else:
        new_mask = sitk.GetArrayFromImage(mask).astype(np.int16)

    for crop_file in glob.glob(os.path.join(input_dir, "*.nii.gz")):
        crop_mask = sitk.ReadImage(crop_file)
        crop_mask = resampler.Execute(crop_mask)

        new_mask += sitk.GetArrayViewFromImage(crop_mask)

    new_mask[new_mask > 0] = 1
    new_mask = sitk.GetImageFromArray(new_mask)
    new_mask.CopyInformation(ref_img)

    return new_mask

def analyze_mask_and_grow(mask, lung_mask, threshold=0.05):
    global _rows, _cols, _levels

    rows = _rows
    cols = _cols
    levels = _levels

    np_mask = sitk.GetArrayViewFromImage(mask)
    np_lung_mask = sitk.GetArrayViewFromImage(lung_mask)

    lung_lower, lung_upper = get_lung_corners(np_lung_mask)
    lung_size = lung_upper - lung_lower

    neighbor_cells = set()
    mask_cells = set()

    for l in range(levels):
        l_begin = lung_upper[0]-1 - (l+1)*lung_size[0]//levels
        l_end = lung_upper[0]-1 - l*lung_size[0]//levels
        level_label = chr(ord('V') + l)

        for r in range(rows):
            r_begin = lung_lower[1] + r*lung_size[1]//rows
            r_end = lung_lower[1] + (r+1)*lung_size[1]//rows
            row_label = chr(ord('A') + r)

            for c in range(cols):
                c_begin = lung_lower[2] + c*lung_size[2]//cols
                c_end = lung_lower[2] + (c+1)*lung_size[2]//cols
                col_label = str(c+1)

                cell_label = row_label + col_label + level_label

                if (np_mask[l_begin:l_end, r_begin:r_end, c_begin:c_end] > 0).any():
                    mask_cells.add(cell_label)

                # XXX: Organized by cell_label format
                # Edge areas
                areas = [ 
                    (c_end-c_begin)*(l_end-l_begin), 
                    (r_end-r_begin)*(l_end-l_begin),
                    (r_end-r_begin)*(c_end-c_begin),
                ]

                # Edge voxel counts
                counts = [ 
                    lambda d : (np_mask[l_begin:l_end, (r_begin if d < 0 else (r_end-1)), c_begin:c_end] > 0).sum(),
                    lambda d : (np_mask[l_begin:l_end, r_begin:r_end, (c_begin if d < 0 else (c_end-1))] > 0).sum(),
                    lambda d : (np_mask[((l_end-1) if d < 0 else l_begin), r_begin:r_end, c_begin:c_end] > 0).sum(), 
                ]

                # Check edges for growth
                for cell_label2 in cell_neighbors(cell_label, connectivity=6):
                    # Find first dimension not in common with cell_label
                    i = [ a == b for a, b in zip(cell_label, cell_label2) ].index(False)
                    diff = ord(cell_label2[i]) - ord(cell_label[i])


                    if counts[i](diff) > 0:
                        print(f"{cell_label} -- {cell_label2} surface area ratio: {counts[i](diff)/areas[i]} ({threshold})")

                    if counts[i](diff) > threshold*areas[i]:
                        neighbor_cells.add(cell_label2)

    #return list(neighbor_cells - mask_cells)
    #mask_cells.update(neighbor_cells)
    grow_cells = set(mask_cells)
    grow_cells.update(neighbor_cells)
    return list(mask_cells), list(grow_cells)

def extract_cells_and_save_images(image_path, lung_mask_path, cells, output_dir):
    global _rows, _cols, _levels

    rows = _rows
    cols = _cols
    levels = _levels

    assert all((isinstance(cell,str) and len(cell) == 3 and cell[0].lower() in "abcde" and cell[1] in "12345" and cell[2].lower() in "vwxyz" for cell in cells))

    cell_ccs = cell_connected_components(cells)

    image_orig = sitk.ReadImage(image_path, sitk.sitkInt16)
    lung_mask_orig = sitk.ReadImage(lung_mask_path)

    image = resample(image_orig, new_spacing=[0.7,0.7,-1], interp="bspline")
    lung_mask = resample(lung_mask_orig, new_spacing=[0.7,0.7,-1], interp="nearest")

    np_image = sitk.GetArrayViewFromImage(image)
    np_lung_mask = sitk.GetArrayViewFromImage(lung_mask)

    lung_lower, lung_upper = get_lung_corners(np_lung_mask)
    lung_size = lung_upper - lung_lower

    os.makedirs(output_dir, exist_ok=True)

    for i, cell_cc in enumerate(cell_ccs):
        image_lower = np.array(np_image.shape, dtype=int)
        image_upper = np.array([0,0,0], dtype=int)

        assert len(cell_cc) > 0

        print(f"CC{i}: {cell_cc}")

        for cell in cell_cc:
            r = ord(cell[0].lower()) - ord('a')
            c = int(cell[1]) - 1
            l = ord(cell[2].lower()) - ord('v')

            r_begin = lung_lower[1] + r*lung_size[1]//rows
            r_end = lung_lower[1] + (r+1)*lung_size[1]//rows

            c_begin = lung_lower[2] + c*lung_size[2]//cols
            c_end = lung_lower[2] + (c+1)*lung_size[2]//cols

            l_begin = lung_upper[0]-1 - (l+1)*lung_size[0]//levels
            l_end = lung_upper[0]-1 - l*lung_size[0]//levels

            image_lower = np.minimum([l_begin, r_begin, c_begin], image_lower)
            image_upper = np.maximum([l_end, r_end, c_end], image_upper)

        cc_image = np_image[image_lower[0]:image_upper[0], image_lower[1]:image_upper[1], image_lower[2]:image_upper[2]]
        new_origin = image.TransformIndexToPhysicalPoint([int(image_lower[2]), int(image_lower[1]), int(image_lower[0])])

        cc_image = sitk.GetImageFromArray(cc_image)
        cc_image.SetSpacing(image.GetSpacing())
        cc_image.SetDirection(image.GetDirection())
        cc_image.SetOrigin(new_origin)

        #output_path=os.path.join(output_dir, f"PICAI_{i:04}_0000.nii.gz")
        output_path=os.path.join(output_dir, f"TGMB_{i:04}_0000.nii.gz")
        print(f"Info: Writing '{output_path}' ...")
        sitk.WriteImage(cc_image, output_path, useCompression=True)

    return image_orig, lung_mask_orig

def draw_cells_exhaustive(image, lung_mask):
    global _rows, _cols, _levels

    color=4095
    line_thickness=2
    text_thickness=1
    font = cv2.FONT_HERSHEY_SIMPLEX

    rows = _rows
    cols = _cols
    levels = _levels
       
    np_image = sitk.GetArrayFromImage(image)
    np_lung_mask = sitk.GetArrayViewFromImage(lung_mask)

    lung_lower, lung_upper = get_lung_corners(np_lung_mask)
    lung_size = lung_upper - lung_lower

    for z in range(lung_lower[0],lung_upper[0]):
        np_slice = np_image[z, ...]

        l = levels*(lung_upper[0]-1-z)//lung_size[0]
        level_label = chr(ord('V') + l)

        for r in range(rows+1):
            r_begin = lung_lower[1] + r*lung_size[1]//rows
            r_end = r_begin

            c_begin = lung_lower[2]
            c_end = lung_upper[2]

            cv2.line(np_slice, (c_begin,r_begin), (c_end,r_end), color, line_thickness)

        for c in range(cols+1):
            c_begin = lung_lower[2] + c*lung_size[2]//cols
            c_end = c_begin

            r_begin = lung_lower[1]
            r_end = lung_upper[1]

            cv2.line(np_slice, (c_begin,r_begin), (c_end,r_end), color, line_thickness)

        for r in range(rows):
            row_label = chr(ord('A') + r)
            r_begin = lung_lower[1] + r*lung_size[1]//rows
            r_end = lung_lower[1] + (r+1)*lung_size[1]//rows
            #r_begin = lung_upper[1] - (r+1)*lung_size[1]//rows
            #r_end = lung_upper[1] - r*lung_size[1]//rows

            cell_height = r_end-r_begin
            text_height = round(0.4*cell_height)

            font_scale = cv2.getFontScaleFromHeight(font, text_height, text_thickness)

            for c in range(cols):
                col_label = str(c+1)
                cell_label = row_label + col_label + level_label
                c_begin = lung_lower[2] + c*lung_size[2]//cols
                c_end = lung_lower[2] + (c+1)*lung_size[2]//cols

                cell_width = c_end - c_begin

                size, baseline = cv2.getTextSize(cell_label, font, font_scale, text_thickness)
                #baseline += text_thickness

                c_label = int(c_begin + (cell_width - size[0])//2)
                r_label = int(r_end - size[1]//2 - baseline)

                cv2.putText(np_slice, cell_label, (c_label, r_label), font, font_scale, color, text_thickness, cv2.LINE_AA)


    new_image = sitk.GetImageFromArray(np_image)
    new_image.CopyInformation(image)

    for key in image.GetMetaDataKeys():
        new_image.SetMetaData(key, image.GetMetaData(key))

    upper = (new_image.GetSize()[0]*0.025, new_image.GetSize()[1]*0.95)
    new_image = DrawForResearchUseOnly(new_image, upper, color=color)

    return new_image

def draw_cells(image, lung_mask):
    global _rows, _cols, _levels

    color=4095
    line_thickness=2
    text_thickness=1
    font = cv2.FONT_HERSHEY_SIMPLEX

    rows = _rows
    cols = _cols
    levels = _levels
       
    np_image = sitk.GetArrayFromImage(image)
    np_lung_mask = sitk.GetArrayViewFromImage(lung_mask)

    lung_lower, lung_upper = get_lung_corners(np_lung_mask)
    lung_size = lung_upper - lung_lower

    level_text_pos = (int(0.05*image.GetSize()[0]), int(0.05*image.GetSize()[1]))
    level_text_height = round(0.4*lung_size[1]//rows)

    #description_text_pos = (level_text_pos[0], level_text_pos[1] + level_text_height + int(0.01*image.GetSize()[0]))
    #description_text_height = level_text_height

    for z in range(lung_lower[0],lung_upper[0]):
        np_slice = np_image[z, ...]

        l = levels*(lung_upper[0]-1-z)//lung_size[0]
        level_label = chr(ord('V') + l)

        level_text = f"Level: {level_label}"
        font_scale = cv2.getFontScaleFromHeight(font, level_text_height, text_thickness)
        cv2.putText(np_slice, level_text, level_text_pos, font, font_scale, color, text_thickness, cv2.LINE_AA)

        for r in range(rows+1):
            r_begin = lung_lower[1] + r*lung_size[1]//rows
            r_end = r_begin

            c_begin = lung_lower[2]
            c_end = lung_upper[2]

            cv2.line(np_slice, (c_begin,r_begin), (c_end,r_end), color, line_thickness)

        for c in range(cols+1):
            c_begin = lung_lower[2] + c*lung_size[2]//cols
            c_end = c_begin

            r_begin = lung_lower[1]
            r_end = lung_upper[1]

            cv2.line(np_slice, (c_begin,r_begin), (c_end,r_end), color, line_thickness)

        for r in range(rows):
            row_label = chr(ord('A') + r)
            r_begin = lung_lower[1] + r*lung_size[1]//rows
            r_end = lung_lower[1] + (r+1)*lung_size[1]//rows

            cell_width = lung_size[2]//cols
            cell_height = r_end - r_begin

            text_height = round(0.4*cell_height)

            c_begin = lung_lower[2] - cell_width
            c_end = lung_lower[2]

            font_scale = cv2.getFontScaleFromHeight(font, text_height, text_thickness)
            size, baseline = cv2.getTextSize(row_label, font, font_scale, text_thickness)

            c_label = int(c_begin + (cell_width - size[0])//2)
            r_label = int(r_end - size[1]//2 - baseline)

            cv2.putText(np_slice, row_label, (c_label, r_label), font, font_scale, color, text_thickness, cv2.LINE_AA)

        for c in range(cols):
            col_label = str(c+1)
            c_begin = lung_lower[2] + c*lung_size[2]//cols
            c_end = lung_lower[2] + (c+1)*lung_size[2]//cols

            cell_width = c_end - c_begin
            cell_height = lung_size[1]//rows

            text_height = round(0.4*cell_height)

            r_begin = lung_lower[1] - cell_height
            r_end = lung_lower[1]
            
            font_scale = cv2.getFontScaleFromHeight(font, text_height, text_thickness)
            size, baseline = cv2.getTextSize(col_label, font, font_scale, text_thickness)

            c_label = int(c_begin + (cell_width - size[0])//2)
            r_label = int(r_end - size[1]//2 - baseline)

            cv2.putText(np_slice, col_label, (c_label, r_label), font, font_scale, color, text_thickness, cv2.LINE_AA)

    new_image = sitk.GetImageFromArray(np_image)
    new_image.CopyInformation(image)

    for key in image.GetMetaDataKeys():
        new_image.SetMetaData(key, image.GetMetaData(key))

    upper = (new_image.GetSize()[0]*0.025, new_image.GetSize()[1]*0.95)
    new_image = DrawForResearchUseOnly(new_image, upper, color=color)

    return new_image

def tabulate_cells(image_dir, mask_dir, lung_mask_dir=None, output_file="cells.csv"):
    global _rows, _cols, _levels

    rows, cols, levels = _rows, _cols, _levels

    row_map = dict()

    for image_file in glob.glob(os.path.join(image_dir, "*.nii.gz")):
        base=os.path.basename(image_file)
        patient_id = base.split("_")[0]

        print(f"Info: Processing '{base}' ...")

        mask_file = os.path.join(mask_dir, base)

        if not os.path.exists(mask_file):
            print(f"Info: Tumor mask file '{mask_file}' missing. Skipping ...")
            continue

        if lung_mask_dir is not None:
            lung_mask_file = os.path.join(lung_mask_dir, base)
            if not os.path.exists(lung_mask_file):
                print(f"Info: Lung mask file '{lung_mask_file}' missing. Skipping ...")
                continue
        else:
            lung_mask_file="lungs.nii.gz"
            segment_lungs_and_others(image_file, lung_mask_file)

        #image = sitk.ReadImage(image_file)
        mask = load_image(mask_file)
        lung_mask = load_image(lung_mask_file)

        #np_image = sitk.GetArrayViewFromImage(image)
        np_mask = sitk.GetArrayViewFromImage(mask)
        np_lung_mask = sitk.GetArrayViewFromImage(lung_mask)

        assert np_mask.shape == np_lung_mask.shape

        lung_lower, lung_upper = get_lung_corners(np_lung_mask)
        lung_size = lung_upper - lung_lower

        for l in range(levels):
            l_begin = lung_upper[0]-1 - (l+1)*lung_size[0]//levels
            l_end = lung_upper[0]-1 - l*lung_size[0]//levels
            level_label = chr(ord('V') + l)

            for r in range(rows):
                r_begin = lung_lower[1] + r*lung_size[1]//rows
                r_end = lung_lower[1] + (r+1)*lung_size[1]//rows
                row_label = chr(ord('A') + r)

                for c in range(cols):
                    c_begin = lung_lower[2] + c*lung_size[2]//cols
                    c_end = lung_lower[2] + (c+1)*lung_size[2]//cols
                    col_label = str(c+1)

                    cell_label = row_label + col_label + level_label

                    if (np_mask[l_begin:l_end, r_begin:r_end, c_begin:c_end] > 0).any():
                        row_map.setdefault(patient_id, set()).add(cell_label)

    print(f"Info: Writing '{output_file}' ...")
    save_cell_map(row_map, output_file)

    print("Done.")

def run_inference(image_path, patient_id=None, cell_map_file="cells.csv", cells=None, convert_image=True, mask=None, segment_lungs=True, work_dir="."):
    cell_map = load_cell_map(cell_map_file)

    meta_data = get_meta_data(image_path)

    os.makedirs(work_dir, exist_ok=True)

    if patient_id is None:
        assert meta_data is not None
        patient_id = meta_data.GetMetaData("0010|0020").strip()
        print(f"Info: Deduced patient ID from DICOM: '{patient_id}'")

    if os.path.isdir(image_path):
        new_image_path=os.path.join(work_dir, "input_image.nii.gz")

        if convert_image or not os.path.exists(new_image_path):
            print(f"Info: Converting to NIFTI...", flush=True)
            image = load_image(image_path)
            save_image(image, new_image_path, compress=True)
        else:
            print(f"Info: Skipping NIFTI conversion...", flush=True)

        image_path = new_image_path

    input_dir = os.path.join(work_dir, "CellImages")
    shutil.rmtree(input_dir, ignore_errors=True)

    lung_mask_path=os.path.join(work_dir, "lungs.nii.gz")

    if segment_lungs or not os.path.exists(lung_mask_path):
        segment_lungs_and_others(image_path, output_path=lung_mask_path)

    if patient_id not in cell_map:
        print(f"Error: '{patient_id}' is not in the cell CSV.")
        image = load_image(image_path)
        lung_mask = load_image(lung_mask_path)
        
        # Don't forget to copy DICOM meta data!
        if meta_data is not None:
            for key in meta_data.GetMetaDataKeys():
                image.SetMetaData(key, meta_data.GetMetaData(key))

        return image, lung_mask

    if cells is None:
        cells = cell_map[patient_id]

    image, lung_mask = extract_cells_and_save_images(image_path, lung_mask_path, cells, input_dir)

    output_dir = os.path.join(work_dir, "CellSegmentations")
    shutil.rmtree(output_dir, ignore_errors=True)

    os.makedirs(output_dir, exist_ok=True)

    run_nnunet(input_dir, output_dir)

    mask = combine_crop_masks(image, output_dir, mask=mask)

    # XXX: Disabled
    #mask = suppress_outside_of_lungs(mask, lung_mask)

    print("Done.")

    if meta_data is not None:
        for key in meta_data.GetMetaDataKeys():
            image.SetMetaData(key, meta_data.GetMetaData(key))

    return image, mask, lung_mask

def save_all_images(mask, grid_image=None, volumetric_image=None, surface_image=None, output_dir=".", series_number=5001, compress=True):
    if grid_image is not None:
        if grid_image.HasMetaDataKey("0020|000e"):
            grid_image.SetMetaData("0020|0011", str(series_number+0))
            grid_image.SetMetaData("0008|103e", "Tumor grid")

            grid_path=os.path.join(output_dir, "Tumor_grid")
            shutil.rmtree(grid_path, ignore_errors=True)

            save_image(grid_image, grid_path, compress=compress)
        else:
            grid_path=os.path.join(output_dir, "Tumor_grid.nii.gz")
            save_image(grid_image, grid_path, compress=compress)

    if volumetric_image is not None:
        if volumetric_image.HasMetaDataKey("0020|000e"):
            volumetric_image.SetMetaData("0020|0011", str(series_number+1))
            volumetric_image.SetMetaData("0008|103e", "Rendered volumetric measurements")

            volumetric_image.EraseMetaData("0028|1050") # Window center
            volumetric_image.EraseMetaData("0028|1051") # Window width
            volumetric_image.EraseMetaData("0028|1055") # Window explanation

            volumetric_image.SetMetaData("0028|1052", "0") # Rescale slope
            volumetric_image.SetMetaData("0028|1053", "1") # Rescale intercept
            volumetric_image.SetMetaData("0008|1054", "US") # US - "unspecified"

            volumetric_path = os.path.join(output_dir, "Rendered_volumetric_measurements")
            shutil.rmtree(volumetric_path, ignore_errors=True)

            save_image(volumetric_image, volumetric_path, compress=compress)
        else:
            volumetric_path = os.path.join(output_dir, "Rendered_volumetric_measurements.nii.gz")
            save_image(volumetric_image, volumetric_path, compress=compress)

    if surface_image is not None:
        if surface_image.HasMetaDataKey("0020|000e"):
            surface_image.SetMetaData("0020|0011", str(series_number+2))
            surface_image.SetMetaData("0008|103e", "HU tumor rendering")

            surface_path=os.path.join(output_dir, "HU_tumor_rendering")
            shutil.rmtree(surface_path, ignore_errors=True)

            save_image(surface_image, surface_path, compress=compress)
        else:
            surface_path=os.path.join(output_dir, "HU_surface_legend.nii.gz")
            save_image(surface_image, surface_path, compress=compress)

    if mask is not None:
        mask_path = os.path.join(output_dir, "tumor_segmentation.nii.gz")
        save_image(mask, mask_path, compress=compress)

def main(image_path, cell_map_file, output_dir=".", work_dir=".", patient_id=None):
    mask=None
    grid_image=None
    volumetric_image=None
    surface_image=None
    cells=None
    segment_lungs=True
    convert_image=True

    num_tries=10
    prev_cells = None

    for _ in range(num_tries):
        #results = run_inference(image_path, patient_id=patient_id, cell_map_file=cell_map_file, cells=cells, convert_image=convert_image, mask=mask, segment_lungs=segment_lungs, work_dir=work_dir)
        results = run_inference(image_path, patient_id=patient_id, cell_map_file=cell_map_file, cells=cells, convert_image=convert_image, mask=None, segment_lungs=segment_lungs, work_dir=work_dir)
        convert_image=False

        # Don't segment lungs again
        segment_lungs=False

        if len(results) == 2:
            image, lung_mask = results
            grid_image = draw_cells(image, lung_mask)
            #phantom_image = _RenderPhantomLegendHU(image)
            break
        else:
            image, mask, lung_mask = results

            tmp_cells, cells = analyze_mask_and_grow(mask, lung_mask)

            if prev_cells is None:
                prev_cells = tmp_cells

            #if len(cells) > 0:
            if prev_cells != cells:
                print(f"Info: Expanding cells to: {cells}")
                prev_cells = cells
                continue

            prev_cells = cells

            grid_image = draw_cells(image, lung_mask)
            volumetric_image = RenderMeasurements(image, mask)
            #phantom_image = _RenderPhantomLegendHU(image)
            surface_image = RenderSurfaceHU(image, mask)
            break

    save_all_images(mask, grid_image=grid_image, volumetric_image=volumetric_image, surface_image=surface_image, output_dir=output_dir)

if __name__ == "__main__":
    os.environ.setdefault("nnUNet_raw", "/home/air/layns/Work/Source/Deploy/thymoma_segmentation_package/thymoma_segmentation_package/Customer/models/nnUNet_raw")
    os.environ.setdefault("nnUNet_preprocessed", "/home/air/layns/Work/Source/Deploy/thymoma_segmentation_package/thymoma_segmentation_package/Customer/models/nnUNet_preprocessed")
    os.environ.setdefault("nnUNet_results", "/home/air/layns/Work/Source/Deploy/thymoma_segmentation_package/thymoma_segmentation_package/Customer/models/nnUNet_results")

    parser = argparse.ArgumentParser(description="Thymoma PACS application.")
    parser.add_argument("--output-dir", dest="output_dir", required=False, type=str, default=".", help="Output folder to save images.")
    parser.add_argument("--work-dir", dest="work_dir", required=False, type=str, default=".", help="Work directory for intermediate results.")
    parser.add_argument("--patient-id", dest="patient_id", required=False, type=str, default=None, help="Patient ID for non-DICOM images.")
    parser.add_argument("--map-file", dest="cell_map_file", required=True, type=str, help="Patient ID vs grid cells CSV file.")
    parser.add_argument("image_path", type=str, help="Input image.")

    args = parser.parse_args()

    main(**vars(args))


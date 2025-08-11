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

import os
import uuid
import time
import SimpleITK as sitk
import numpy as np
import cv2
import functools
import operator
from recist import ComputeRECIST

class ColorMap(dict):
    def __init__(self):
        super().__init__({
            1: (255,0,0),
            2: (0,255,0),
            3: (0,255,255),
            4: (255,127,255),
            5: (255,255,0),
            6: (255,128,0),
        })

    def __contains__(self, i):
        return i != 0

    def __getitem__(self, i):
        if i < 0:
            return (0,0,192)

        i = ((i-1) % len(self)) + 1
        return super().__getitem__(i)

def DrawForResearchUseOnly(image, upper, color=(255,255,255)):
    version_text="20240819"
    line_thickness=3
    text_thickness=1
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_label = f"FOR RESEARCH USE ONLY (Ver. {version_text})"

    text_height = round(0.03*image.GetSize()[1])
    font_scale = cv2.getFontScaleFromHeight(font, round(text_height*0.9), text_thickness)

    np_image = sitk.GetArrayFromImage(image)

    xu = int(upper[0])
    yu = int(upper[1])

    size, baseline = cv2.getTextSize(text_label, font, font_scale, text_thickness)
    baseline += text_thickness

    xt = xu
    yt = yu + size[1]

    for z in range(np_image.shape[0]):
        cv2.putText(np_image[z, ...], text_label, (xt, yt), font, font_scale, color, text_thickness, cv2.LINE_AA)

    new_image = sitk.GetImageFromArray(np_image)
    new_image.CopyInformation(image)

    for key in image.GetMetaDataKeys():
        new_image.SetMetaData(key, image.GetMetaData(key))

    return new_image

def _DrawPhantomLegendHU(image, upper, width, height, label_levels=5, label_color=4095):
    text_thickness=1
    font = cv2.FONT_HERSHEY_SIMPLEX

    #text_height = round(0.03*image.GetSize()[1])
    text_height = round(0.3*height)
    font_scale = cv2.getFontScaleFromHeight(font, round(text_height*0.9), text_thickness)

    np_image = sitk.GetArrayFromImage(image)

    xu = upper[0]
    yu = upper[1]

    xl = min(np_image.shape[2], xu + width)
    yl = min(np_image.shape[1], yu + height)

    width = xl - xu
    height = yl - yu

    def color_map(x):
        return int(-1000 + 5095*(x-xu)/(width-1))

    labels = { xu + (l*(width-1))//(label_levels-1): str(color_map(xu + (l*(width-1))//(label_levels-1))) for l in range(label_levels) }

    for z in range(np_image.shape[0]):
        np_slice = np_image[z, ...]

        for x in range(xu,xl):
            color = color_map(x)
            cv2.line(np_slice, (x,yu), (x,yl), color, 1)

            if x in labels:
                text_label = labels[x]
                size, baseline = cv2.getTextSize(text_label, font, font_scale, text_thickness)
                baseline += text_thickness

                xt = x - size[0]//2 # Center the number
                yt = yl + size[1]

                cv2.putText(np_slice, text_label, (xt, yt), font, font_scale, label_color, text_thickness, cv2.LINE_AA)

    new_image = sitk.GetImageFromArray(np_image)
    new_image.CopyInformation(image)

    for key in image.GetMetaDataKeys():
        new_image.SetMetaData(key, image.GetMetaData(key))

    return new_image

def _RenderPhantomLegendHU(image):
    size = image.GetSize()

    width = round(0.7*size[0])
    height = round(0.1*size[1])

    upper = (int(0.15*size[0]), int(0.4*size[1]))

    new_image = _DrawPhantomLegendHU(image, upper, width, height, label_levels=5, label_color=4095)

    return new_image

def RenderSurfaceHU(image, tumor_mask, color=4095):
    np_image = sitk.GetArrayFromImage(image)
    np_tumor_mask = sitk.GetArrayViewFromImage(tumor_mask)

    assert np_image.shape == np_tumor_mask.shape

    np_image[np_tumor_mask > 0] = color

    new_image = sitk.GetImageFromArray(np_image)
    new_image.CopyInformation(image)

    for key in image.GetMetaDataKeys():
        new_image.SetMetaData(key, image.GetMetaData(key))

    return new_image
              
def DrawLegend(image, upper, colors, text_labels, obj_count, footer_label=None, footer_color=(255,255,255), tiny_label=None, tiny_color=(0,0,192)):
    line_thickness=3
    text_thickness=1
    font = cv2.FONT_HERSHEY_SIMPLEX
    #font_scale = 1

    np_image = sitk.GetArrayFromImage(image)

    color_width = round(0.03*image.GetSize()[0])
    color_height = round(0.03*image.GetSize()[1])
    text_gap = round(0.01*image.GetSize()[0])

    font_scale = cv2.getFontScaleFromHeight(font, round(color_height*0.9), text_thickness)

    # NOTE: Avoid a gap in rendered results with this offset value!
    offset = 0

    for i in range(1,obj_count+3):
        is_tiny = (i == obj_count+1)
        is_footer = (i == obj_count+2)

        if is_tiny and tiny_label is None:
            offset -= 1
            continue

        if is_footer and footer_label is None:
            offset -= 1
            continue

        i += offset

        if is_tiny or is_footer:
            color = tiny_color if is_tiny else footer_color
            text_label = tiny_label if is_tiny else footer_label
        else:
            color = colors[i]
            text_label = text_labels[i]

        xu = int(upper[0])
        yu = int(upper[1] + (i-1)*color_height)

        xl = int(xu + color_width)
        yl = int(yu + color_height)

        size, baseline = cv2.getTextSize(text_label, font, font_scale, text_thickness)
        baseline += text_thickness

        #xt = xl + text_gap + size[0]//2
        xt = xl + text_gap
        yt = yu + size[1]

        if not is_footer:
            for z in range(np_image.shape[0]):
                cv2.rectangle(np_image[z, ...], (xu, yu), (xl, yl), color, -1)
                cv2.putText(np_image[z, ...], text_label, (xt, yt), font, font_scale, color, text_thickness, cv2.LINE_AA)
        else:
            for z in range(np_image.shape[0]):
                cv2.putText(np_image[z, ...], "=", (xu, yt), font, font_scale, color, text_thickness, cv2.LINE_AA)
                # NOTE: Sigma is also chr(963).upper()
                #cv2.putText(np_image[z, ...], "Σ", (xu, yt), font, font_scale, color, text_thickness, cv2.LINE_AA)
                cv2.putText(np_image[z, ...], text_label, (xt, yt), font, font_scale, color, text_thickness, cv2.LINE_AA)

    new_image = sitk.GetImageFromArray(np_image)
    new_image.CopyInformation(image)

    for key in image.GetMetaDataKeys():
        new_image.SetMetaData(key, image.GetMetaData(key))

    return new_image

def RelabelSmallComponents(cc_mask, tiny_volume_cm3=1.0, tiny_label=-1):
    spacing = cc_mask.GetSpacing()
    voxel_volume = functools.reduce(operator.mul, spacing) / 1000.0 # cm^3

    print(f"cc_mask spacing: {spacing}")

    np_cc_mask = sitk.GetArrayViewFromImage(cc_mask)
    np_new_cc_mask = np.zeros(np_cc_mask.shape, dtype=np.int16)

    obj_count = np_cc_mask.max()

    tiny_volume_sum = 0.0
    tiny_count = 0
    new_obj_count = 0

    for label in range(1,obj_count+1):
        tumor_mask = (np_cc_mask == label)
        tumor_count = tumor_mask.sum()

        if tumor_count == 0:
            continue

        tumor_volume = tumor_count * voxel_volume

        if tumor_volume < tiny_volume_cm3:
            np_new_cc_mask[tumor_mask] = tiny_label
            tiny_volume_sum += tumor_volume
            tiny_count += 1
        else:
            new_obj_count += 1
            np_new_cc_mask[tumor_mask] = new_obj_count

    new_cc_mask = sitk.GetImageFromArray(np_new_cc_mask)
    new_cc_mask.CopyInformation(cc_mask)

    return new_cc_mask, new_obj_count, tiny_count, tiny_volume_sum

def RenderMeasurements(image, tumor_mask):
    tiny_volume_cm3 = 1.0

    #colors = { 1: (255, 0, 0) }
    colors = ColorMap()

    spacing = image.GetSpacing()

    voxel_volume = functools.reduce(operator.mul, spacing) / 1000.0 # cm^3

    image = ColorizeImage(WindowImage(image))

    small = round(0.1/voxel_volume)

    #np_cc_mask, obj_count = ConnectedComponents(sitk.GetArrayFromImage(tumor_mask))
    np_cc_mask, obj_count = ConnectedComponents(RemoveSmallConnectedComponents(sitk.GetArrayViewFromImage(tumor_mask), small=small))

    #for label in range(1, obj_count):
    #    if (np_cc_mask == label).sum() < 15:
    #        np_cc_mask[np_cc_mask == label] = 0


    cc_mask = sitk.GetImageFromArray(np_cc_mask)
    cc_mask.CopyInformation(tumor_mask)

    cc_mask, obj_count, tiny_count, tiny_volume_sum = RelabelSmallComponents(cc_mask, tiny_volume_cm3=tiny_volume_cm3, tiny_label=-1)
    np_cc_mask = sitk.GetArrayFromImage(cc_mask)

    major_axes, minor_axes = ComputeRECIST(cc_mask, obj_count)

    print(f"Info: CC object count = {obj_count}")

    #tumor_mask, blend_mask = ColorizeMask(tumor_mask, colors)
    cc_mask, blend_mask = ColorizeMask(cc_mask, colors)

    output_image = Blend(image, cc_mask, blend_mask, alpha=0.5)

    output_image, recist_labels = RenderRECIST(output_image, major_axes, minor_axes)

    text_labels = dict()

    tumor_burden = tiny_volume_sum

    for label in range(1, obj_count+1):
        tumor_mask = (np_cc_mask == label)
        tumor_count = tumor_mask.sum()
        tumor_volume = tumor_count * voxel_volume
        tumor_burden += tumor_volume

        text_labels[label] = f"{tumor_volume:.2f} cc"

        if label in recist_labels:
            text_labels[label] += f", {recist_labels[label]}"

        #tumor_indices = np.argwhere(tumor_mask)
        #lower = tumor_indices.min(axis=0)
        #upper = tumor_indices.max(axis=0)

    if tiny_count > 0:
        tiny_label=f"<{tiny_volume_cm3:.2f} cc, sum = {tiny_volume_sum:.2f} cc, count = {tiny_count}"
    else:
        tiny_label=None
        
    footer_label=f"{tumor_burden:.2f} cc"
        
    upper = (image.GetSize()[0]*0.025, image.GetSize()[1]*0.025)
    output_image = DrawLegend(output_image, upper, colors, text_labels, obj_count, footer_label=footer_label, tiny_label=tiny_label, tiny_color=colors[-1])

    upper = (image.GetSize()[0]*0.025, image.GetSize()[1]*0.95)
    output_image = DrawForResearchUseOnly(output_image, upper)

    return output_image

def RenderRECIST(image, major_axes, minor_axes, major_color=(255,255,255), minor_color=(0,0,0), line_thickness=1):
    text_labels = dict()

    np_image = sitk.GetArrayFromImage(image)

    for label, (u,v,z,length) in major_axes.items():
        np_slice = np_image[z, ...]
        length_cm = length / 10.0

        u = ( int(u[0]), int(u[1]) )
        v = ( int(v[0]), int(v[1]) )

        cv2.line(np_slice, u, v, major_color, line_thickness)

        text_labels[label] = f"maj = {length_cm:.2f} cm"

    for label, (u,v,z,length) in minor_axes.items():
        np_slice = np_image[z, ...]
        length_cm = length / 10.0

        u = ( int(u[0]), int(u[1]) )
        v = ( int(v[0]), int(v[1]) )

        cv2.line(np_slice, u, v, minor_color, line_thickness)

        if label in text_labels:
            text_labels[label] += f", min = {length_cm:.2f} cm"
        else:
            text_labels[label] = f"min = {length_cm:.2f} cm"

    new_image = sitk.GetImageFromArray(np_image, isVector=True)
    new_image.CopyInformation(image)

    for key in image.GetMetaDataKeys():
        new_image.SetMetaData(key, image.GetMetaData(key))

    return new_image, text_labels


def ConnectedComponents(npMask):
    ccFilter = sitk.ConnectedComponentImageFilter()
    ccFilter.SetFullyConnected(True)

    ccMask = ccFilter.Execute(sitk.GetImageFromArray(npMask))
    objCount = ccFilter.GetObjectCount()

    return sitk.GetArrayFromImage(ccMask), objCount

def RemoveSmallConnectedComponents(npMask, small=15):
    npMask = npMask.copy()
    npCcMask, objCount = ConnectedComponents(npMask)

    for label in range(1, objCount+1):
        npObjMask = (npCcMask == label)

        if npObjMask.sum() < small:
            npMask[npObjMask] = 0

    return npMask

def WindowImage(img):
    level = 50
    width = 400
    np_img = sitk.GetArrayViewFromImage(img)

    lower = level - 0.5*width

    np_img = np.clip(255*(np_img - lower)/width, 0, 255).astype(np.uint8)

    new_img = sitk.GetImageFromArray(np_img)
    new_img.CopyInformation(img)

    for key in img.GetMetaDataKeys():
        new_img.SetMetaData(key, img.GetMetaData(key))

    return new_img

def ColorizeImage(img):
    if img.GetNumberOfComponentsPerPixel() == 3:
        return img
    
    if img.GetNumberOfComponentsPerPixel() != 1:
        raise RuntimeError("Don't know how to colorize non-grayscale images.")
        
    npImg = sitk.GetArrayViewFromImage(img)
    
    # Window image
    minValue, maxValue = np.percentile(npImg, [1, 99])
    npImg = 255*(npImg - minValue)/(maxValue - minValue)
    npImg = np.clip(npImg, 0, 255).astype(np.uint8)
    
    # Duplicate grayscale channel for R, G and B
    npImg = np.repeat(npImg[..., None], 3, axis=-1) # R=G=B in colorized grayscale images
    
    newImg = sitk.GetImageFromArray(npImg, isVector=True)
    
    newImg.CopyInformation(img)
    
    # Copy DICOM tags!
    for key in img.GetMetaDataKeys():
        newImg.SetMetaData(key, img.GetMetaData(key))
        
    return newImg

def ColorizeMask(mask, colors):
    if mask.GetNumberOfComponentsPerPixel() == 3:
        return mask
        
    if mask.GetNumberOfComponentsPerPixel() != 1:
        raise RuntimeError("Don't know how to colorize non-grayscale masks.")

    npMask = sitk.GetArrayViewFromImage(mask)
    
    npNewMask = np.zeros(list(npMask.shape) + [3], dtype=np.uint8)
    npBlendMask = np.zeros(npMask.shape, dtype=np.uint8)
    
    # Get unique labels from mask and fill them in with colors
    for label in np.unique(npMask[npMask != 0]):
        if label not in colors:
            continue
            
        npBlendMask += (npMask == label)
        
        # There's probably a more efficient way to do this
        for c, color in enumerate(colors[label]):
            npNewMask[..., c] += ((npMask == label)*color).astype(np.uint8)
            
    newMask = sitk.GetImageFromArray(npNewMask, isVector=True)    
    newMask.CopyInformation(mask)
    
    blendMask = sitk.GetImageFromArray(npBlendMask) # Be consistent with returns

    return newMask, blendMask

def Blend(image, mask, blendMask, alpha=0.5):
    npImage = sitk.GetArrayFromImage(image) # NOTE: Not "ArrayView"
    npMask = sitk.GetArrayViewFromImage(mask)
    npBlendMask = sitk.GetArrayViewFromImage(blendMask)
    
    npNewImage = npImage
    npNewImage[npBlendMask != 0] = (1.0 - alpha)*npImage[npBlendMask != 0] + alpha*npMask[npBlendMask != 0]
    
    newImage = sitk.GetImageFromArray(npNewImage.astype(np.uint8))
    
    newImage.CopyInformation(image)
    
    # Copy DICOM tags!
    for key in image.GetMetaDataKeys():
        newImage.SetMetaData(key, image.GetMetaData(key))
        
    return newImage

def _RemoveBadSlices(fileNames):
    if len(fileNames) == 0:
        return fileNames

    reader = sitk.ImageFileReader()
    reader.SetImageIO("GDCMImageIO")

    def GetSize(path):
        try:
            reader.SetFileName(path)
            reader.ReadImageInformation()
            return reader.GetSize()
        except:
            return None

    totalSlices = len(fileNames)
    maxBadSlices = max(1, int(0.25*totalSlices))

    fileSizes = [ (fileName, GetSize(fileName)) for fileName in fileNames ]
    fileSizes = [ pair for pair in fileSizes if pair[1] is not None ]

    assert totalSlices - len(fileSizes) <= maxBadSlices, f"Too many bad slices! Failed to read {totalSlices - len(fileSizes)} slices."

    # Take median file size as the correct size
    tmp = sorted(fileSizes, key=lambda pair : functools.reduce(operator.mul, pair[1]))
    medSize = tmp[len(tmp)//2][1]

    newFileNames = [ fileName for fileName, size in fileSizes if size == medSize ]

    assert totalSlices - len(newFileNames) <= maxBadSlices, f"Too many bad slices! There were {totalSlices - len(newFileNames)} inconsistently-sized slices."

    return newFileNames

def LoadDicomImage(path, seriesUID = None, dim = None, dtype = None):
    if not os.path.exists(path):
        return None

    reader2D = sitk.ImageFileReader()
    reader2D.SetImageIO("GDCMImageIO")
    reader2D.SetLoadPrivateTags(True)

    if dtype is not None:
        reader2D.SetOutputPixelType(dtype)

    if dim is None: # Guess the dimension by the path
        dim = 2 if os.path.isfile(path) else 3

    if dim == 2:
        reader2D.SetFileName(path)

        try:
            return reader2D.Execute()
        except:
            return None

    if os.path.isfile(path):
        reader2D.SetFileName(path)

        try:
            reader2D.ReadImageInformation()
            seriesUID = reader2D.GetMetaData("0020|000e").strip()
        except:
            return None

        path = os.path.dirname(path)
        
    fileNames = []

    if seriesUID is None or seriesUID == "":
        allSeriesUIDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path)

        if len(allSeriesUIDs) == 0:
            return None

        for tmpUID in allSeriesUIDs:
            tmpFileNames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path, tmpUID)        
            
            if len(tmpFileNames) > len(fileNames):
                seriesUID = tmpUID
                fileNames = tmpFileNames # Take largest series
    else:
        fileNames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path, seriesUID)

    oldLen = len(fileNames)

    fileNames = _RemoveBadSlices(fileNames)

    if oldLen != len(fileNames):
        print(f"Info: Removed {oldLen - len(fileNames)} slices.")

    if len(fileNames) == 0: # Huh?
        return None

    reader3D = sitk.ImageSeriesReader()
    reader3D.SetImageIO("GDCMImageIO")
    reader3D.SetFileNames(fileNames)
    reader3D.SetLoadPrivateTags(True)
    reader3D.SetMetaDataDictionaryArrayUpdate(True)

    #reader3D.SetOutputPixelType(sitk.sitkUInt16)

    if dtype is not None:
        reader3D.SetOutputPixelType(dtype)

    try:
        image = reader3D.Execute()
    except:
        return None

    # Check if meta data is available!
    # Copy it if it is not!
    if not image.HasMetaDataKey("0020|000e"):
        for key in reader3D.GetMetaDataKeys(1):
            image.SetMetaData(key, reader3D.GetMetaData(1, key))

    return image

def SaveDicomImage(image, path, compress=True):
    # Implement pydicom's behavior
    def GenerateUID(prefix="1.2.826.0.1.3680043.8.498."):
        if not prefix:
            prefix = "2.25."
        
        uid = str(prefix) + str(uuid.uuid4().int)
        return uid[:64] # Prevent error with UIDs longer than 64 characters

    if image.GetDimension() != 2 and image.GetDimension() != 3:
        raise RuntimeError("Only 2D or 3D images are supported.")

    if not image.HasMetaDataKey("0020|000e"):
        print("Error: Reference meta data does not appear to be DICOM?", file=sys.stderr)
        return False

    writer = sitk.ImageFileWriter()
    writer.SetImageIO("GDCMImageIO")
    writer.SetKeepOriginalImageUID(True)
    writer.SetUseCompression(compress)

    newSeriesUID = GenerateUID()

    if image.GetDimension() == 2:
        writer.SetFileName(path)

        imageSlice = sitk.Image([image.GetSize()[0], image.GetSize()[1], 1], image.GetPixelID(), image.GetNumberOfComponentsPerPixel())
        imageSlice.SetSpacing(image.GetSpacing())

        imageSlice[:,:,0] = image[:]

        # Copy meta data
        for key in image.GetMetaDataKeys():
            imageSlice.SetMetaData(key, image.GetMetaData(key))

        newSopInstanceUID = GenerateUID()

        imageSlice.SetMetaData("0020|000e", newSeriesUID)
        imageSlice.SetMetaData("0008|0018", newSopInstanceUID)
        imageSlice.SetMetaData("0008|0003", newSopInstanceUID)

        try:
            writer.Execute(imageSlice)
        except:
            return False

        return True

    if not os.path.exists(path):
        os.makedirs(path)

    for z in range(image.GetDepth()):
        newSopInstanceUID = GenerateUID()

        imageSlice = sitk.Image([image.GetSize()[0], image.GetSize()[1], 1], image.GetPixelID(), image.GetNumberOfComponentsPerPixel())

        imageSlice[:] = image[:,:,z]
        imageSlice.SetSpacing(image.GetSpacing())

        # Copy meta data
        for key in image.GetMetaDataKeys():
            imageSlice.SetMetaData(key, image.GetMetaData(key))

        # Then write new meta data ...
        imageSlice.SetMetaData("0020|000e", newSeriesUID)
        imageSlice.SetMetaData("0008|0018", newSopInstanceUID)
        imageSlice.SetMetaData("0008|0003", newSopInstanceUID)

        # Instance creation date and time
        imageSlice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
        imageSlice.SetMetaData("0008|0013", time.strftime("%H%M%S"))

        # Image number
        imageSlice.SetMetaData("0020|0013", str(z+1))

        position = image.TransformIndexToPhysicalPoint((0,0,z))

        # Image position patient
        imageSlice.SetMetaData("0020|0032", f"{position[0]}\\{position[1]}\\{position[2]}")

        # Slice location
        imageSlice.SetMetaData("0020|1041", str(position[2]))

        # Spacing
        imageSlice.SetMetaData("0018|0050", str(image.GetSpacing()[2]))
        imageSlice.SetMetaData("0018|0088", str(image.GetSpacing()[2]))

        imageSlice.EraseMetaData("0028|0106")
        imageSlice.EraseMetaData("0028|0107")

        slicePath = os.path.join(path, f"{z+1}.dcm")
        writer.SetFileName(slicePath)

        try:
            writer.Execute(imageSlice)
        except:
            print(f"Error: Failed to write slice '{slicePath}'.", file=sys.stderr)
            return False

    return True



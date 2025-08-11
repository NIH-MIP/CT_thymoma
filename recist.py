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

import SimpleITK as sitk
import numpy as np
import cv2

def ComputeRECIST(cc_mask, obj_count=None):
    tol=1e-2

    spacing = np.array(list(cc_mask.GetSpacing()))

    print(f"Info: Spacing = {spacing}")

    np_cc_mask = sitk.GetArrayViewFromImage(cc_mask)

    if obj_count is None:
        obj_count = np_cc_max.max()

    major_axes = dict()
    minor_axes = dict()

    for label in range(1, obj_count+1):
        np_mask = (np_cc_mask == label).astype(np.uint8)

        indices = np.argwhere(np_mask)
        lower = indices.min(axis=0)
        upper = indices.max(axis=0)

        major_axis = None
        minor_axis = None

        for z in range(lower[0], upper[0]+1):
            contours = cv2.findContours(np_mask[z, ...], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            contours = contours[0] if len(contours) == 2 else contours[1] # XXX: Workaround for different versions of OpenCV

            contours = [ contour.squeeze(axis=1) for contour in contours if contour.shape[0] > 2 ]

            for contour in contours:
                U, S, Vh = np.linalg.svd(np.cov((contour*spacing[:2]).T))

                assert U.shape[0] == 2 and U.shape[1] == 2

                #print(f"CC{label}: U = {U}, S = {S}, Vh = {Vh}")

                for i in range(contour.shape[0]):
                    rawU = contour[i,:]
                    u = rawU*spacing[:2]

                    for j in range(i+1, contour.shape[0]):
                        rawV = contour[j,:]
                        v = rawV*spacing[:2]
    
                        direction = (u-v)
                        length = np.linalg.norm(direction)

                        if length < 1e-5:
                            continue

                        direction /= length

                        if 1.0 - abs(np.inner(direction, U[:,0])) < tol:
                            if major_axis is None or length > major_axis[3]:
                                major_axis = (rawU,rawV,z,length)

                        if 1.0 - abs(np.inner(direction, U[:,1])) < tol:
                            if minor_axis is None or length > minor_axis[3]:
                                minor_axis = (rawU,rawV,z,length)


        if major_axis is not None:
            major_axes[label] = major_axis

        if minor_axis is not None:
            minor_axes[label] = minor_axis

    return major_axes, minor_axes


#problems with this code: the rotation of dapi is fixed to match specific HE image, HE is normalized over all channels insteaed of separately(?) and who knows what happens when images are not square, but this problem is silently fixed by the affine step for near-square images

import SimpleITK as sitk
import tifffile
import zarr
import logging
import shutil
import numpy as np
from skimage.transform import resize

logging.getLogger('tifffile').setLevel(logging.ERROR)


from skimage.color import rgb2hed

def color_deconvolve_(he_path):
    with open(he_path, "rb") as f:
        mag = f.read(8)
    if mag[:6] == b'\x93NUMPY':
        HE_arr = np.load(he_path)
    elif mag[:2] in (b'II', b'MM'):
        with tifffile.TiffFile(he_path) as tif:
            HE_arr = tif.asarray()
    else:
        raise ValueError(f"unrecognized format: {he_path}")
    hed = rgb2hed(HE_arr)
    return [hed[..., 0], hed[..., 1]]

  
def img_arr_(img_path):
    with open(img_path, "rb") as f:
        mag = f.read(8)
    if mag[:6] == b'\x93NUMPY':
        dapi_arr = np.load(img_path)
    elif mag[:2] in (b'II', b'MM'):
        with tifffile.TiffFile(img_path) as tif:
            dapi_arr = tif.asarray()
    else:
        raise ValueError(f"unrecognized format: {dapi_arr}")
    return dapi_arr

def save_tif_(img_path, img_arr):
  tifffile.imwrite(img_path, img_arr)

def print_iteration_(reg_method):
  print(f"L{reg_method.GetCurrentLevel()} "
        f"{reg_method.GetOptimizerIteration():3} = "
        f"{reg_method.GetMetricValue():.6f}")

def align(HE_path, dapi_path):
    [H, E] = color_deconvolve_(HE_path)
    H = (H - H.min())/(H.max() - H.min())
    E = (E - E.min())/(E.max() - E.min())

    HE_full = img_arr_(HE_path)
    HE_full = (HE_full - HE_full.min())/(HE_full.max() - HE_full.min()) #should this be normalized per channel?

    dapi = img_arr_(dapi_path)
    dapi = np.rot90(dapi, k=-1)
    dapi = (dapi - dapi.min())/(dapi.max() - dapi.min())
    
    H = resize(H, dapi.shape, anti_aliasing=True)
    HE_full = resize(HE_full, (dapi.shape[0], dapi.shape[1], 3), anti_aliasing=True)##

    fixed = sitk.GetImageFromArray(dapi.astype(np.float32))
    moving = sitk.GetImageFromArray(H.astype(np.float32))

    HE_full_uint8 = (HE_full * 255).clip(0, 255).astype(np.uint8)
    moving_fullHE = sitk.GetImageFromArray(HE_full_uint8, isVector=True)

    mask = sitk.TriangleThreshold(fixed, 0, 1)

    closed = sitk.BinaryMorphologicalClosing(mask, [10, 10])
    opened = sitk.BinaryMorphologicalOpening(closed, [10, 10])
    mask_fixed = sitk.BinaryFillhole(opened)

    reg = reg_init_()
    transf_rigid = rigid_euler_(reg, fixed, moving)#each method returns the concatenated transforms up to that point, with the latest one contatenated to all the previous ones
    transf_affine = affine_(reg, fixed, moving, transf_rigid)
    transf_bspline = nonlinear1_(reg, fixed, moving, transf_affine) 

    registered_H = sitk.Resample(moving, fixed, transf_bspline, sitk.sitkLinear, 0.0, moving.GetPixelID())

    total_transf = patch_refinement_bspline_(fixed, moving, mask_fixed, registered_H, transf_bspline)

    registered_H = sitk.Resample( #register the original moving image
        moving,                 # raw moving, per your outline
        fixed,                  # reference grid
        total_transf,
        sitk.sitkLinear,
        0.0,                    # default pixel value
        moving.GetPixelID())
    
    registered_HE = sitk.Resample(
        moving_fullHE, 
        fixed, 
        total_transf, 
        sitk.sitkLinear, 
        0.0, 
        moving_fullHE.GetPixelID())
    
    return (registered_H, registered_HE)
    

    

def reg_init_():
    reg = sitk.ImageRegistrationMethod()
    #reg.SetMetricFixedMask(mask_fixed)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.05)
    reg.AddCommand(sitk.sitkIterationEvent, lambda: print_iteration_(reg))
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=300,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
        estimateLearningRate=reg.EachIteration)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 2, 1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[4, 2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return reg

def rigid_euler_(reg, fixed, moving):
   #stage 1, Euler2d
    
    reg.SetInitialTransform(sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler2DTransform()
    ))

    transform_rigid = reg.Execute(fixed, moving)

    return transform_rigid

def affine_(reg, fixed, moving, prev_transf):
    reg.SetMovingInitialTransform(prev_transf)#setmovingInitialtransform is the previous transformation you want to use as a starting point
    reg.SetInitialTransform(sitk.AffineTransform(2)) #setinitialtransform is the kind of new transformation you will return

    transform_affine = reg.Execute(fixed, moving)
    
    transform_stage2 = sitk.CompositeTransform([prev_transf, transform_affine])
    return transform_stage2


def nonlinear1_(reg, fixed, moving, prev_transf):
    bspline = sitk.BSplineTransformInitializer(
        image1=fixed,
        transformDomainMeshSize=[8, 8],
    )
    reg.SetMetricSamplingPercentage(0.01)
    reg.SetMovingInitialTransform(prev_transf)#

    reg.SetInitialTransformAsBSpline(bspline, inPlace=True, scaleFactors=[1, 1, 2, 4])
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=12,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
        estimateLearningRate=reg.Once)

    transform_bspline = reg.Execute(fixed, moving)
    transform_bspline_final = sitk.CompositeTransform([prev_transf, transform_bspline]) #the transform gets applied right to left

    return transform_bspline_final

def patch_refinement_bspline_(fixed, moving, mask_fixed, prev_img, prev_transf):
    to_field_transf = sitk.TransformToDisplacementFieldFilter()
    to_field_transf.SetReferenceImage(fixed)
    disp_field = to_field_transf.Execute(prev_transf)
    disp_field_arr = sitk.GetArrayFromImage(disp_field)
    disp_resid_arr = np.zeros(disp_field_arr.shape)
    Wsum = np.zeros(disp_field_arr.shape[:2])

    ref_size = fixed.GetSize()
    W = ref_size[0]
    Ht = ref_size[1]
    patch = 2**8
    overlap = 2**6
    stride = patch - overlap

    R = sitk.ImageRegistrationMethod() #this object is just for computing LNCC
    #R.SetMetricFixedMask(mask_fixed)
    R.SetMetricAsANTSNeighborhoodCorrelation(4)
    R.SetMetricSamplingStrategy(R.NONE)
    R.SetInterpolator(sitk.sitkLinear) #R for calculating local cross correlation
    R.SetInitialTransform(sitk.Transform(2, sitk.sitkIdentity))

    # the registration method for the actual bspline deformation
    reg_deform = sitk.ImageRegistrationMethod()
    reg_deform.SetMetricSamplingPercentage(0.01)
    reg_deform.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=12,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
        estimateLearningRate=reg_deform.Once)
    reg_deform.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg_deform.SetMetricSamplingStrategy(reg_deform.RANDOM)
    reg_deform.SetOptimizerScalesFromPhysicalShift()
    reg_deform.SetInterpolator(sitk.sitkLinear)

    bad_patches = []

    B = np.full((Ht, W), np.nan)

    for y0 in range(0, Ht, stride):
        for x0 in range(0, W, stride):
            x1 = min(x0 + patch, W)
            y1 = min(y0 + patch, Ht)

            x0c = max(0, x1 - patch) #the max is not strictly necessary
            y0c = max(0, y1 - patch)
            #check if the NCC is bad enough to require correction
            patch_mask = sitk.RegionOfInterest(mask_fixed, [x1 - x0c, y1-y0c], [x0c, y0c])

            m = sitk.GetArrayFromImage(patch_mask)
            if m.sum() < (patch**2)*0.1:
                continue

            patch_fixed = sitk.RegionOfInterest(fixed, [x1 - x0c, y1-y0c], [x0c, y0c])
            patch_moving = sitk.RegionOfInterest(prev_img, [x1 - x0c, y1-y0c], [x0c, y0c])

            R.SetMetricFixedMask(patch_mask)
            met = R.MetricEvaluate(patch_fixed, patch_moving)
            if met > 1e300:
                continue


            #B[y0c:y1, x0c:x1] = met

            #threshold for correction
            if met > -0.25:
                B[y0c:y1, x0c:x1] = 1

                #the correction transform
                bspline = sitk.BSplineTransformInitializer(
                    image1=patch_fixed,
                    transformDomainMeshSize=[8, 8],
                )
                reg_deform.SetInitialTransformAsBSpline(bspline, inPlace=True, scaleFactors=[1])

                patch_deform = reg_deform.Execute(patch_fixed, patch_moving)


                patch_displace = sitk.TransformToDisplacementFieldFilter()
                patch_displace.SetReferenceImage(patch_fixed)
                patch_disp_field = patch_displace.Execute(patch_deform)

                patch_disp_field_arr = sitk.GetArrayFromImage(patch_disp_field)


                #falloff
                wy = np.minimum(np.arange(patch), np.arange(patch)[::-1])+1
                wx = np.minimum(np.arange(patch), np.arange(patch)[::-1])+1
                taper = np.outer(wy, wx).astype(float)
                taper = taper/taper.max()


                disp_resid_arr[y0c:y1, x0c:x1] +=  patch_disp_field_arr*taper[..., None]
                Wsum[y0c:y1, x0c:x1] += taper[...]

    disp_resid_arr /= np.maximum(Wsum, 1e-8)[..., None]
    disp_field_arr += disp_resid_arr

    #final resampling:
    disp_img = sitk.GetImageFromArray(disp_field_arr, isVector=True)
    disp_img.CopyInformation(fixed)        # origin/spacing/direction from full grid
    disp_img = sitk.Cast(disp_img, sitk.sitkVectorFloat64)   # transform requires float64

    total_transform = sitk.DisplacementFieldTransform(disp_img) #SHOUDL BE RENAMED TO FINAL TRANSFORM

    return total_transform
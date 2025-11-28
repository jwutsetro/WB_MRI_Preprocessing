import SimpleITK as sitk


def pad_to_ref(original, target, mode=sitk.sitkNearestNeighbor):
    """
               Args:
                   original: The original image. The size of this image will
                   target: The target image. The original image will be padded to fit the same space as the target image
                   mode: Set the interpolator.
                   """
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(target)
    resample.SetInterpolator(mode)

    new_image = resample.Execute(original)
    return new_image

import blosc2
import numpy as np
import SimpleITK as sitk


class NfitiReaderWriter:
    def __init__(self):
        """
        NfitiReaderWriter class constructor. This class is responsible for reading and writing nifti files.
        """

    def read(self, image_path: str):
        """
        Read the image file.

        Returns:
            sitk_image (SimpleITK.Image): The image object.
        """
        sitk_image = sitk.ReadImage(image_path)
        array = sitk.GetArrayFromImage(sitk_image)
        return array, sitk_image

    def write(self, array: np.ndarray, sitk_image: sitk.Image, output_path: str):
        """
        Write the image file.

        Parameters:
            sitk_image (SimpleITK.Image): The image object.
            output_path (str): The path to save the image file.
        """
        out_image = sitk.GetImageFromArray(array)
        out_image.SetDirection(sitk_image.GetDirection())
        out_image.SetOrigin(sitk_image.GetOrigin())
        out_image.SetSpacing(sitk_image.GetSpacing())
        sitk.WriteImage(out_image, output_path)


class BloscReaderWriter:
    def __init__(self):
        """
        BloscReaderWriter class constructor. This class is responsible for reading and writing blosc files.
        """
        blosc2.set_nthreads(1)

    def read(self, image_path: str):
        """
        Read the image file.

        Returns:
            array (np.ndarray): The image array.
        """
        dparams = {"nthreads": 1}
        im = blosc2.open(urlpath=image_path, mode="r", dparams=dparams, mmap_mode="r")
        return im[:], None

    def write(self, array: np.ndarray, properties, output_path: str):
        """
        Write the image file.

        Parameters:
            array (np.ndarray): The image array.
            properties: Unused
            output_path (str): The path to save the image file.
        """
        cparams = {
            "codec": blosc2.Codec.ZSTD,
            # 'filters': [blosc2.Filter.SHUFFLE],
            # 'splitmode': blosc2.SplitMode.ALWAYS_SPLIT,
            "clevel": 8,
        }
        blosc2.asarray(
            np.ascontiguousarray(array),
            urlpath=output_path,
            chunks=None,
            blocks=None,
            cparams=cparams,
            mmap_mode="w+",
        )

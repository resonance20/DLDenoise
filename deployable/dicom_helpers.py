import numpy as np
import pydicom as dicom
import os
import uuid

#%%Read dicom folder
def read_dicom_folder(folder):

    list_files = os.listdir(folder)
    list_dicom = [file for file in list_files if file.endswith('.dcm')]

    ref = dicom.dcmread(os.path.join(folder, list_dicom[0]))
    array = np.zeros((len(list_dicom), ref.Rows, ref.Columns), dtype=np.uint16)

    for i, file in enumerate(sorted(list_dicom)):
        array[i] = dicom.dcmread(os.path.join(folder, file)).pixel_array

    return array

#%%Update uuid
def _update_uuid(old_uuid, new_uuid):
    split_uuid = old_uuid.split('.')
    split_uuid[-1] = str(new_uuid)
    new_uuid_str = ""
    for s in split_uuid:
        new_uuid_str += "."+s
    return new_uuid_str[1:]

#%%Write dicom folder
def write_dicom_folder(folder, new_volume, output_folder, series_description):

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    new_volume = new_volume.astype(np.uint16)

    series_uid=uuid.uuid4().int

    list_files = os.listdir(folder)
    list_dicom = [file for file in list_files if file.endswith('.dcm')]

    assert len(list_dicom)==new_volume.shape[0]

    for i, file in enumerate(sorted(list_dicom)):
        
        sop_uid=uuid.uuid4().int
        dic_file = dicom.dcmread(os.path.join(folder, file))

        try:
            assert dic_file.Rows==new_volume.shape[1]
            assert dic_file.Columns==new_volume.shape[2]
        except AssertionError:
            assert dic_file.Rows==(new_volume.shape[1]*2)
            assert dic_file.Columns==(new_volume.shape[2]*2)
            dic_file.Rows = new_volume.shape[1]
            dic_file.Columns = new_volume.shape[2]
            spacings = dic_file.PixelSpacing
            for spac in range(len(spacings)):
                spacings[spac] *= 2
            dic_file.PixelSpacing = spacings

        dic_file.ImageType[0] = 'DERIVED'
        dic_file.SeriesInstanceUID = _update_uuid(dic_file.SeriesInstanceUID, series_uid)
        dic_file.SOPInstanceUID = _update_uuid(dic_file.SOPInstanceUID, sop_uid)
        dic_file.file_meta.MediaStorageSOPInstanceUID = dic_file.SOPInstanceUID
        dic_file.SeriesDescription = series_description

        dic_file.PixelData = new_volume[i].tobytes()
        dic_file.LargestImagePixelValue = int(np.max(new_volume[i]))
        dic_file.SmallestImagePixelValue = int(np.min(new_volume[i]))
        
        dic_file.save_as(os.path.join(output_folder, file))

    return None
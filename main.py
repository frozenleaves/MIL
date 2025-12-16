import os
from config import Paths, WSIConfig
from wsi_processor import extract_wsi_features

svs_file = "/home/frozen/Medical_Info_Classification/A_Datasets/pathology/patient_001/slide1.svs"

extract_wsi_features(svs_file, Paths.WSI_FEAT_DIR, WSIConfig())
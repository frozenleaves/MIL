from mil.offline_feature_prepare import generate_index_file  as gif
from mil.offline_feature_prepare import genereate_wsi_features as gwf
from mil.utils import create_symlink_split, extract_txt_from_doc 



def doc2txt(data_dir):
    extract_txt_from_doc(data_dir)


def prepare_dataset(source_dir, target_dir, train_ratio=0.5, seed=42):
    create_symlink_split(source_dir, target_dir, train_ratio=train_ratio, seed=seed)


def generate_wsi_features(overwrite=False):
    gwf(overwrite=overwrite)


def generate_index_file(extract_features=False, overwrite=False):
    gif(extract_features=extract_features, overwrite=overwrite)

if __name__ == "__main__":
    # doc2txt("/media/codingma/LLM/data-1005")
    #generate_wsi_features(overwrite=False)
    #source_dir = "/media/codingma/LLM/data-1005"
    #target_dir = "/media/codingma/LLM/lcx/Medical_Info_Classification/datasets"
    #create_symlink_split(source_dir, target_dir, train_ratio=0.5, seed=42)
    generate_index_file()


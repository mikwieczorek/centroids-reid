# Inference

## Creating embeddings
Example command to create embeddings using pretrained model.

Path to images is controlled by  `DATASETS.ROOT_DIR` and may be path to a single image file or to directory, which contains valid images.  
Additional flag can be set `--images-in-subfolders` – only images in subfolders in the `DATASETS.ROOT_DIR` will be used.ag can be set `--images-in-subfolders` – only images in subfolders in the `DATASETS.ROOT_DIR` will be used.

**NOTE:** at the beggining of the script, there is `exctract_func` specified, which is used to extracy `pair_id` either from filename or from subfolder name. See the examples in the script and customize if needed.

```
#from repo root folder
bash test_scripts/embed.sh
```

## Running similarity serach
Example command to run similarity search using pretrained model and query images.  
Path to images is controlled by  `DATASETS.ROOT_DIR` and may be path to a single image file or to directory, which contains valid images.   
Additional flag can be set `--images-in-subfolders` – only images in subfolders in the `DATASETS.ROOT_DIR` will be used.

```
#from repo root folder
bash test_scripts/similar.sh
```
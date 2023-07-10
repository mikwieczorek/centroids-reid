import cv2
import numpy as np

results = np.load("similarity-outputs/results.npy", allow_pickle=True)

results = results.item()

for query_path, result in results.items():

    # combine two images into one and save
    img1 = cv2.imread(query_path)
    img2 = cv2.imread(result["paths"][0])

    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    img = np.concatenate((img1, img2), axis=1)

    cv2.imwrite(f"tmp/{result['distances'][0]:.03}.png", img)

    print(query_path, result["paths"][0])

    print(result)


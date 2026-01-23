import cv2
import numpy as np  

def occlusion_xai(
        model,
        image,
        window_size=32,
        stride=16,
        top_k=5
):
    """
    Occlusion-based XAI using sliding window + blur
    Returns top-k most influential regions
    """

    h, w, _ = image.shape

    image = image / 255.0

    original_pred = model.predict(image[np.newaxis, ...], verbose=0)[0][0]
    results = []

    for y in range(0, h - window_size, stride):
        for x in range(0, w - window_size, stride):

            occluded_image = image.copy()

            patch = occluded_image[y:y+window_size, x:x+window_size]
            blurred = cv2.GaussianBlur(patch, (15, 15), 0)
            occluded_image[y:y+window_size, x:x+window_size] = blurred

            pred = model.predict(
                occluded_image[np.newaxis, ...],
                verbose=0
            )[0][0]

            confidence_drop = original_pred - pred

            results.append({
                "x": x,
                "y": y,
                "drop": float(confidence_drop)
            })

    results = sorted(results, key=lambda x: x["drop"], reverse=True)
    return results[:top_k], float(original_pred)
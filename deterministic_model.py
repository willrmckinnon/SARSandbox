import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt




def mean_threshold_model(sample, show_checkpoints = False, alpha = 2):
    # 1. Setting a brightness threshold using avergage and stanadard deviation
    std = np.std(sample.gray)
    avg = np.average(sample.gray)
    threshold = avg + (alpha*std)

    if show_checkpoints:
        print(f"Sample Stats:\n   - Mean: {avg}\n   - Standard Dev: {std}\n   - Threshold: {threshold}\n")


    # 2. Identify anomoly pixels and filter to only show large groups
    anomalies = sample.gray > threshold
    labels, n_groups = ndimage.label(
        anomalies,
        structure=np.ones((3, 3), dtype=bool),
    )

    if show_checkpoints: 
        plt.imshow(anomalies)
        plt.title(fr'Anomoly Pixels above {alpha}$sigma$ above Mean')
        plt.axis('off')
        plt.show()



    # 3. Find each group's bounding box.
    regions = ndimage.find_objects(labels)

    keep_labels = []
    boxes = []

    for label_id, region in enumerate(regions, start=1):
        if region is None:
            continue

        rows, cols = region

        height = rows.stop - rows.start
        width = cols.stop - cols.start

        if height >= 10 and width >= 10:
            keep_labels.append(label_id)

            # Same pixel-coordinate format as sample.boxes_xyxy.
            boxes.append([
                cols.start, rows.start,
                cols.stop, rows.stop,
            ])

    # 4. Retain only bright pixels belonging to qualifying groups.
    filtered_anomalies = np.isin(labels, keep_labels)

    # Candidate detection boxes for later evaluation.
    predicted_boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)

    if show_checkpoints: print(f'{len(predicted_boxes)} Total Boxes Identified')


    return predicted_boxes







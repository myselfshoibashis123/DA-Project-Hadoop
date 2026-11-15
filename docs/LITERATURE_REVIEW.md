# Literature Review and References

> **Verification note.** This project was built in an environment without web-search
> access. Items marked **[verified]** were downloaded and used directly while building the
> project. Items marked **[from memory]** are well-known publications cited from the
> author's background knowledge; their bibliographic details (volume, pages, year) should
> be checked against Google Scholar / the publisher before submission.

## 1. Computer vision for construction and industrial safety

Manual safety inspection is periodic and labour-intensive, so a large body of work since
about 2017 uses camera footage and deep-learning object detectors to monitor compliance
automatically. Three themes dominate:

**PPE compliance detection.** Early work focused on hard hats. Fang et al. (2018) used a
Faster R-CNN detector to find workers *not* wearing hard hats in far-field surveillance
video **[from memory]**. Wu et al. (2019) released the GDUT-HWD benchmark and studied
hard-hat detection with colour classes **[from memory]**. Nath, Behzadan and Paal (2020)
compared three YOLO-based strategies for detecting workers, hard hats and vests, and
released the Pictor-PPE dataset **[from memory]**. Their finding that a single detector
trained on combined "worker + PPE" classes is a practical choice for real-time use
influenced the single-model design adopted here.

**Explicit negative classes.** A recurring problem is that the *absence* of a detected
helmet does not prove a violation: the head may be occluded or out of frame. Datasets that
label "no-helmet"/"no-vest" directly (such as the Construction-PPE dataset used here,
which has `no_helmet`, `no_vest`, `no_gloves`, `no_goggle`, `no_boots`) let a model detect
violations positively. This project therefore treats only explicit negative detections as
violations by default and records everything else as `unknown`; the stricter "absence =
violation" policy is available as a configuration option.

**Manufacturing-specific data.** Most public PPE datasets come from construction sites.
SH17 (Ahmad & Rahimi, 2024) targets manufacturing environments with 17 classes
**[from memory]** and is a natural next dataset for this project (see "Future work" in
the README).

**Hazardous zones and proximity.** Restricted-area intrusion is commonly implemented as a
geometric rule on top of person detection (point-in-polygon on the worker's ground contact
point), and worker-equipment proximity as a distance threshold between detected boxes. This
project follows that rule-based approach rather than claiming learned activity or accident
recognition, which would require temporal models and labelled incident video that are not
publicly available at suitable scale.

## 2. Object detection: the YOLO family

YOLO (Redmon et al., 2016) reframed detection as a single regression pass, making
real-time detection practical **[from memory]**. The Ultralytics implementation
(YOLOv8, 2023; YOLO11, 2024) provides COCO-pretrained checkpoints and a training API
**[verified: package `ultralytics` 8.4.x and `yolo11n.pt` used in this project]**.
Transfer learning from COCO (Lin et al., 2014) **[from memory]** is the standard way to
obtain usable PPE detectors from a few hundred labelled images, which is what
`scripts/train_ppe_model.py` does.

## 3. Big-data processing with Hadoop

MapReduce (Dean & Ghemawat, 2004) expresses batch aggregation as map, shuffle/sort and
reduce phases **[from memory]**; HDFS (Shvachko et al., 2010) provides replicated block
storage for large files such as video **[from memory]**. In video analytics pipelines,
Hadoop is typically used for (a) durable storage of raw footage and derived records and
(b) batch aggregation of detection logs, while GPU inference runs outside MapReduce. This
project follows that split: inference produces JSON-lines records, and Hadoop Streaming
(`hadoop/mapper.py`, `hadoop/reducer.py`) aggregates them.

## 4. Datasets considered

| Dataset | Classes (relevant) | Licence | Used? |
|---|---|---|---|
| **Ultralytics Construction-PPE** – `https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip`, docs `https://docs.ultralytics.com/datasets/detect/construction-ppe` | helmet, vest, gloves, boots, goggles, Person, forklift, and 5 `no_*` violation classes; 800 train / 200 val images | CC BY 4.0 | **Yes [verified]** |
| SH17 (manufacturing PPE) | 17 classes incl. person, helmet, gloves, safety-vest, glasses… | see dataset page | No – recommended extension **[from memory]** |
| Pictor-PPE (Nath et al.) | worker, hat, vest | see repository | No **[from memory]** |
| GDUT-HWD (Wu et al.) | hard hats by colour, person | see paper | No **[from memory]** |
| COCO | person (used via pretrained weights) | CC BY 4.0 (annotations) | Indirectly (pretrained `yolo11n.pt`) **[verified]** |

Construction-PPE was chosen because it is (1) downloadable from a stable public URL with
no sign-up, (2) already in YOLO format, (3) the only one of these with explicit negative
classes for five PPE items and a forklift class, and (4) small enough to fine-tune on a CPU.

## 5. References

1. Ultralytics. *Construction-PPE dataset* (CC BY 4.0). https://docs.ultralytics.com/datasets/detect/construction-ppe **[verified]**
2. Jocher, G., Qiu, J., Chaurasia, A. *Ultralytics YOLO* (software). https://github.com/ultralytics/ultralytics **[verified]**
3. Redmon, J., Divvala, S., Girshick, R., Farhadi, A. "You Only Look Once: Unified, Real-Time Object Detection." CVPR 2016. **[from memory]**
4. Lin, T.-Y. et al. "Microsoft COCO: Common Objects in Context." ECCV 2014. **[from memory]**
5. Fang, Q., Li, H., Luo, X., Ding, L., Luo, H., Rose, T. M., An, W. "Detecting non-hardhat-use by a deep learning method from far-field surveillance videos." *Automation in Construction* 85 (2018). **[from memory]**
6. Wu, J., Cai, N., Chen, W., Wang, H., Wang, G. "Automatic detection of hardhats worn by construction personnel: A deep learning approach and benchmark dataset." *Automation in Construction* 106 (2019). **[from memory]**
7. Nath, N. D., Behzadan, A. H., Paal, S. G. "Deep learning for site safety: Real-time detection of personal protective equipment." *Automation in Construction* 112 (2020). **[from memory]**
8. Ahmad, H. M., Rahimi, A. "SH17: A dataset for human safety and personal protective equipment detection in manufacturing industry." arXiv:2407.04590 (2024). **[from memory]**
9. Dean, J., Ghemawat, S. "MapReduce: Simplified Data Processing on Large Clusters." OSDI 2004. **[from memory]**
10. Shvachko, K., Kuang, H., Radia, S., Chansler, R. "The Hadoop Distributed File System." IEEE MSST 2010. **[from memory]**
11. Apache Software Foundation. *Hadoop Streaming* documentation. https://hadoop.apache.org/docs/stable/hadoop-streaming/HadoopStreaming.html **[from memory]**
12. Bradski, G. "The OpenCV Library." *Dr. Dobb's Journal of Software Tools* (2000). **[from memory]**

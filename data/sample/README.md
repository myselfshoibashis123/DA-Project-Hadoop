# Sample data

* `images/` – 8 images from the **held-out test split** of the Ultralytics
  Construction-PPE dataset, resized to max 640 px. They were never used for training.
  Licence: **CC BY 4.0** – source: Ultralytics, https://docs.ultralytics.com/datasets/detect/construction-ppe
* `videos/site_walkthrough.mp4` – a **synthetic slideshow video** generated from those
  images by `scripts/make_sample_video.py` (slow zoom over each still). It exists only to
  demonstrate the video code path; it is **not real CCTV footage**.

To analyse your own footage, pass any image folder or `.mp4` file to
`scripts/run_pipeline.py`.

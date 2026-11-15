# DA-Project-Hadoop

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt.

python scripts/download_weights.py        # COCO person model

bash scripts/run_end_to_end.sh            # pipeline + MapReduce (Hadoop if running, else simulation)

streamlit run app/dashboard.p

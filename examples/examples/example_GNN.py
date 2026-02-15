from sars.gnn_multimodal import GNNMultimodalConfig, run_full_pipeline
import logging

logging.basicConfig(level=logging.INFO)

config = GNNMultimodalConfig()
results = run_full_pipeline(config, atlases=["synthseg", "schaefer_100", "aal3", "brainnetome"])

import logging, sys, torch, glob, re
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

from segger.training.segger_data_module import SeggerDataModule
from segger.models.segger_model import Segger
from segger.training.train import LitSegger
from torch_geometric.nn import to_hetero
from segger.prediction.predict_parquet import segment
from pathlib import Path

dm = SeggerDataModule(
    data_dir='/data-hdd0/Ethan_Baird/Dec25_xenium/segger_tiles',
    batch_size=1, num_workers=2,
)
dm.setup()
logger.info(f"Data loaded: {len(dm.train)} train tiles")

model_path = Path('/data-hdd0/Ethan_Baird/Dec25_xenium/segger_model/lightning_logs/version_5')
ckpt_dir = model_path / 'checkpoints'
ckpts = sorted(glob.glob(str(ckpt_dir / '*.ckpt')),
               key=lambda c: tuple(int(x) for x in re.findall(r'\d+', c)))
ckpt_path = ckpts[-1]
logger.info(f"Using checkpoint: {ckpt_path}")

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state = ckpt['state_dict']
num_tx_tokens = state['model.tx_embedding.tx.weight'].shape[0]
init_emb      = state['model.tx_embedding.tx.weight'].shape[1]
first_att_key = [k for k in state if 'conv_first' in k and k.endswith('.att')][0]
_, heads, hid_ch = state[first_att_key].shape
last_att_key  = [k for k in state if 'conv_last'  in k and k.endswith('.att')][0]
_, _,     out_ch = state[last_att_key].shape
num_mid = len(set(k.split('.conv_mid_layers.')[1].split('.')[0]
                  for k in state if '.conv_mid_layers.' in k))

base_model = Segger(num_tx_tokens=num_tx_tokens, init_emb=init_emb,
                    hidden_channels=hid_ch, out_channels=out_ch,
                    heads=heads, num_mid_layers=num_mid)
metadata = dm.train[0].metadata()
hetero_model = to_hetero(base_model, metadata=metadata, aggr='sum')
lit = LitSegger(model=hetero_model, learning_rate=1e-3)
lit.load_state_dict(state)
lit.eval()
logger.info("Model reconstructed OK")

segment(
    lit, dm,
    save_dir='/data-hdd0/Ethan_Baird/Dec25_xenium/segger_output',
    seg_tag='segger_dec25',
    transcript_file='/data-hdd0/Ethan_Baird/Dec25_xenium/outs_subset/transcripts.parquet',
    receptive_field={'k_bd': 4, 'dist_bd': 12.0, 'k_tx': 5, 'dist_tx': 5.0},
    use_cc=False,
    knn_method='kd_tree',
    verbose=True,
)
logger.info("DONE")

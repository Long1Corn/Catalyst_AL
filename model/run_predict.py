from cgcnn.cfg import CFG
from cgcnn.main import Crystal_Trainer
import warnings


if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    cfg = CFG()

    # cfg.label_dir = r"D:\Pyprojects\catalyst\AL\Cat_Data\Slab_All"
    # cfg.structure_dir = r"D:\Pyprojects\catalyst\AL\Cat_Data\Slab_All"
    cfg.label_dir = r"AL/Cat_Data/R2"
    cfg.structure_dir = r"AL/Cat_Data/Raw_Crystal_Data_All"

    # cfg.model_path = "D:\Pyprojects\catalyst\AL\cgcnn-master\cgcnn\save_dir\Round_1\model.pth"

    cfg.properties = ["Binding_energy", "Energy_barrier"]

    cfg.test_ratio = 0.2

    cfg.augmentation = True

    cfg.epochs = 400
    cfg.lr_milestones = [300, ]
    cfg.lr = 0.01
    cfg.batch_size = 8
    cfg.weight_decay = 1e-4
    cfg.dropout = 0.1

    cfg.n_conv = 3
    cfg.n_h = 2
    cfg.h_fea_len = 64
    cfg.atom_fea_len = 128
    cfg.out_size = 2

    cfg.print_freq = 5
    cfg.test_freq = 500

    cfg.save_dir = 'test'

    crystal_model = Crystal_Trainer(cfg)
    crystal_model.train()

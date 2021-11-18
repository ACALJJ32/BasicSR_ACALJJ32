import torch
from basicsr.archs.edvr_arch import EDVR


def load_model():
    model_config = dict(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_frame=5,
        deformable_groups=8,
        num_extract_block=5,
        num_reconstruct_block=10,
        center_frame_idx=None,
        hr_in=False,
        with_predeblur=False,
        with_tsa=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() != 0 else 'cpu')

    edvr_model = EDVR(**model_config).to(device)

    edvr_net = torch.jit.trace(edvr_model, torch.rand(1, 5, 3, 64, 64).cuda())

    edvr_net.save("edvr-m.pt")

def load_trace_model(model_path):
    model = torch.jit.load(model_path)
    return model

if __name__ == "__main__":
    model_path = "./edvr-m.pt"
    # load_model()

    img = torch.rand((1,5,3,64,64)).cuda()

    device = torch.device('cuda' if torch.cuda.is_available() != 0 else 'cpu')

    model = load_trace_model(model_path)
    model = model.to(device)

    output = model(img)
    print(output.size())

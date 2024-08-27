import matplotlib.pyplot as plt
from torchcam.methods import GradCAM, CAM, LayerCAM, ScoreCAM
from torchcam.utils import overlay_mask
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image

def display_cam(model, img_path):
    cam_extractor = LayerCAM(model)
    img = read_image(img_path)

    input_tensor = resize(img, [448, 448]) / 255
    input_tensor = normalize(input_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input_tensor = input_tensor.unsqueeze(0)
    device = model.parameters().__next__().device
    input_tensor = input_tensor.to(device)
    #print(input_tensor.shape)

    out = model(input_tensor)
    out.cpu()
    #print(out.shape)
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    #print(activation_map[0].shape)
    #print(activation_map[0].squeeze(0).shape)

    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    plt.imshow(result)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # plt.imshow(activation_map[0].squeeze(0).numpy())
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()
import torchvision.models.mobilenetv2
from huggingface_hub import list_models

from utils_common import *
import matplotlib.pyplot as plt

from torchcam.methods import GradCAM, CAM, LayerCAM, ScoreCAM
from torchcam.utils import overlay_mask

from torchvision.io.image import read_image
from torchvision import transforms
# from skimage.transform import resize
from torchvision.transforms.functional import normalize, resize, to_pil_image
import timm
from cam_heatmap import *

data, train_loader, test_loader = load_images('./images', batch_size=16, shuffle=True)

num_classes = len(data.classes)
aaa = timm.list_models('*efficientnet*')
model = timm.create_model('efficientnet_b5', num_classes=num_classes)
# model = torchvision.models.shufflenet_v2_x1_0(weights=None)

optimizer = optim.AdamW(model.parameters(), lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.045, momentum=0.9, weight_decay=0.00004)

model_name = 'efficientnet'
# num_epochs = 9
# train_losses, train_accs = train_model(model, train_loader, optimizer, num_epochs)
# torch.save(model.state_dict(), f'{model_name}.pth')
# test_report(model, f'{model_name}.pth', test_loader, data)
# eval_model(model, test_loader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(f'{model_name}.pth', weights_only=True))
# model.to(device)
# test_report(model, f'{model_name}.pth', test_loader, data)
# eval_model(model, test_loader)

display_cam(model, 'images/1/crop_G8V2416001S1VHN51_143351860.jpg')

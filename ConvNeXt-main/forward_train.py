import torch
from timm.models import create_model
import torch.cuda.profiler as ncu

import models.convnext
import models.convnext_isotropic

batch_size = 64
image_size = 224
num_classes = 25

model = create_model("convnext_tiny", pretrained=False, num_classes=num_classes)
model = model.cuda().train()

inputs = torch.randn(batch_size, 3, image_size, image_size).cuda()

torch.cuda.synchronize()
ncu.start()

with torch.cuda.amp.autocast():
    outputs = model(inputs)

ncu.stop()
torch.cuda.synchronize()

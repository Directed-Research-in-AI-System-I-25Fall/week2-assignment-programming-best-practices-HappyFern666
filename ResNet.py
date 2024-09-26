import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, ResNetForImageClassification
from torchvision import transforms
from torch.utils.data import DataLoader

# 加载 MNIST 数据集
mnist = load_dataset("dataset/mnist")

# 定义图像处理器 (Image Preprocessor) 和模型
image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

print(model)
num_parameters = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {num_parameters}")

# (28，28)->(224,224), make it to Tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.Grayscale(3),        # 将单通道灰度图像扩展为3通道
    transforms.ToTensor()           # 转换为Tensor
])

def preprocess_data(examples):
    images = [transform(image) for image in examples['image']]
    return {"image": images, "label": examples['label']}

mnist.set_transform(preprocess_data)
dataloader = DataLoader(mnist['test'], batch_size=16)

print("1")

correct = 0
total = 0

print(len(dataloader))

i = 0

model.eval()  #set the model to evaluation mode
with torch.no_grad(): 
    for batch in dataloader:

        inputs = image_processor(batch['image'], return_tensors="pt")

        outputs = model(**inputs).logits
        predicted_labels = outputs.argmax(dim=-1)  
        print(predicted_labels)
        print(batch['label'])

        correct += (predicted_labels == torch.tensor(batch['label'])).sum().item()
        total += len(batch['label'])
        i += 1
        print(i, correct, total)

# 计算最终的准确率
accuracy = correct / total
print(f"Accuracy on MNIST test dataset: {accuracy:.4f}")

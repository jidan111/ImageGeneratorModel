# ImageGeneratorModel
用于学习的图片生成模型实现，包含GAN,VAE,Diffusion  
部分模型结果：
<!-- 第一行 -->
<div style="display: flex; justify-content: center; margin-bottom: 10px;">
  <img src="./result/dcgan.png" width="256" alt="dcgan" style="margin-right: 10px;"/>
  <img src="./result/ddpm.png" width="256" alt="ddpm" style="margin-right: 10px;"/>
  <img src="./result/ddim.png" width="256" alt="ddim"/>
</div>

<!-- 第二行 -->
<div style="display: flex; justify-content: center;">
  <img src="./result/vae.png" width="256" alt="vae" style="margin-right: 10px;"/>
  <img src="./result/ddpm2.png" width="256" alt="aapm2"/>
</div>

代码框架：
		./LinLanDeepLearningFrame/  
		  ├── ImageGeneratorModel/  
		  │   ├── BaseStruct.py  
		  │   ├── DiffusionModel.py  
		  │   ├── GANModel.py  
		  │   ├── Losses.py  
		  │   ├── UnetModel.py  
		  │   ├── VAEModel.py  
		  │   ├── __init__.py  
		  │   ├── functions.py  
		  │   └── utils.py  
		  ├── Tokenizr/  
		  │   ├── BPE.py  
		  │   └── __init__.py  
		  ├── TransformerModels/  
		  │   ├── BaseStruct.py  
		  │   ├── Transformer.py  
		  │   └── __init__.py  
		  └── __init__.py  
在LinLanDeepLearningFrame.ImageGeneratorModel.utils里面包含了各个模型的训练函数，支持混合精度训练



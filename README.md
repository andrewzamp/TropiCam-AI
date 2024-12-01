<div align="center">

# 🌴 **TropiCam-AI**  

![banner](./assets/readme_banner.jpg)  

Advancing Arboreal Wildlife Monitoring in the Neotropics

---

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Demo-fcc624)](https://huggingface.co/spaces/andrewzamp/TropiCam-demo)

</div

### 📜 **Overview**

Surveying arboreal wildlife in tropical rainforests has long been a challenging endeavor. **TropiCam-AI** is the first machine learning-based solution specifically designed to automate the classification of **Neotropical arboreal mammals and birds** from camera-trap images.  

Built on the cutting-edge **ConvNeXt architecture**, TropiCam-AI provides unparalleled accuracy, flexibility, and ease of use for researchers and conservationists working to preserve biodiversity in the world's most ecologically diverse regions.  

### Key features  
- 🐒 Automated classification of **50+ arboreal species**.  
- 📊 High accuracy: **~90% at species level**, **99% at class level**.  
- 🌐 Trained on **200,000+ images** from Brazil, Peru, Costa Rica, and French Guiana.  
- 🚀 Easy-to-implement for researchers without programming experience.  

---

## 🖥️ **Installation**  

### Prerequisites  
- [Miniforge](https://github.com/conda-forge/miniforge "Install Miniforge") or [Miniconda](https://docs.anaconda.com/miniconda/install/ "Install Miniconda") 
- [git](https://git-scm.com/downloads "Install git")
- Additional dependencies will be downloaded from `environment.yaml`

### Quick start
Open a Miniforge or Anaconda command prompt (depending on whether you installed Miniforge or Miniconda), then type:
```bash  
# Clone the repository  
git clone https://github.com/andrewzamp/TropiCam-AI.git  
cd TropiCam-AI  

# Create and activate the environment  
conda create -f environment.yaml -y
conda activate TropiCam-AI

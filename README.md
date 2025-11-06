<div align="center">

# 🌴 **TropiCam-AI**  

![banner](./assets/readme_banner.jpg)  

Advancing Arboreal Wildlife Monitoring in the Neotropics

---

[![Publication](https://img.shields.io/badge/Publication-Paper-D95C5C)](https://huggingface.co/spaces/andrewzamp/TropiCam-demo)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Demo-fcc624)](https://huggingface.co/spaces/andrewzamp/TropiCam-demo)

</div

Surveying arboreal wildlife in tropical rainforests has long been a challenging endeavor. **TropiCam-AI** is the first machine learning-based solution specifically designed to automate the classification of Neotropical arboreal mammals and birds from camera-trap images and videos. It was released with the publication "**Introducing TropiCam-AI: A taxonomically flexible automated classifier of Neotropical arboreal mammals and birds from camera-trap data**" (URL).

Built on a ConvNeXt architecture and trained on both an extensive camera-trapping dataset and citize-science images, TropiCam-AI can recognize a 84 taxa of arboreal mammals and birds, and it is optimized with a taxonomic aggregation strategy to return classification at multiple levels of the taxonomy hierarchy (e.g. species, genus, family, class and order) to enhance accuracy, flexibility and generalizability to unseen locations and species. TropiCam-AI aims to provide accuracy, flexibility, and ease of use for researchers and conservationists working to preserve biodiversity in one of the world's most biodiverse regions.  

---

## How to use it? 
### AddaxAI  
We wanted to make **TropiCam-AI** as easy to use as possible, without needing extensive knowledge on how machine learning algorithms work or coding/programming skills. So, our model is available on the AddaxAI platform, a user-friendly and code-free application that allows deployment of machine learning classifiers with the simple click of a button. User can directly install AddaxAI from the official website, and then select our model from the zoo of available algorithms. After classification is finished, results can be made available in a number of different formats: among the most relevant, we highlight CSV/XLSX for direct integration in subsequent analyses, or the default JSON file that can be integrated in the popular **Timelapse** software for image visualization. 

A **few tips**:
- TropiCam-AI can be used the classic way, forcing detected animals to be classified as one of the 84 taxa we used to train it (see here the full list). However, the AddaxAI platform also provides a **taxonomic aggregation** feature, where if an animal is identified with low certainty, classification is returned at the higher taxonomic level (e.g. from species to genus). The model keeps iterating this procedure until it reaches the required confidence: this allows TropiCam-AI to still provide useful information instead of forcing classification at the species level under challenging scenarios.
- We set the aggregation threshold at **0.75** by default, which means that every classification with confidence value lower than 0.75 will be aggregated at a higher taxonomy. This value was determined empirically to maximize accuracy and taxonomic resolution, and we provide more details in our publication. However, we encourage users to experiment based on their needs: a lower threshold means that the model will return more classifications at species level, even the most uncertain ones; a higher threshold will result in a more fleible approach, where difficult images will be classified at a higher taxonomy more often.
- The AddaxAI platform also provides a number of post-inference utilities (manually annotate results, separating images in folders based on classification output, exporting classification results in CSV or XLSX, prodicing plots, and others). We refer to the official website and GitHub repository for more information.

![image](./assets/AddaxAI_screenshot.png) 

### Single-image demo 
We also provide a very simple demo, that can be used to test TropiCam-AI on single images on the cloud. Note: images uploaded to the demo are used only for inference and are NOT retained, temporarily or permanently, by us or the Gradio platform.

![image](./assets/Gradio_demo_screenshot.png) 

---

## How to cite us
...

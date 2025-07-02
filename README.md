# UPDA-main

This is the official project  repository containing the implementation Code for UPDA: Unsupervised Progressive Domain Adaptation for No-Reference Point Cloud Quality Assessment by Bingxu Xie, Jincan Wu, Yonghui Liu, Weiqing Li, and Zhiyong Su

Our paper addresses the issue of the distribution shift across various point cloud quality databases and proposes methods to to improve the generalization performance of existing PCQA models.

![image-20250702113844589](.\question.png)

### **Datasets**

 SJTU-PCQA: [Predicting the Perceptual Quality of Point Cloud: A 3D-to-2D Projection-Based Exploration](https://ieeexplore.ieee.org/abstract/document/9238424) | [Link](https://smt.sjtu.edu.cn/database/)                                                                            

WPC: [Perceptual Quality Assessment of Colored 3D Point Clouds](https://ieeexplore.ieee.org/document/9756929)  | [Link](https://github.com/qdushl/Waterloo-Point-Cloud-Database)  

WPC2.0(Compression): [Reduced Reference Perceptual Quality Model with Application to Rate Control for Video-based Point Cloud Compression](https://ieeexplore.ieee.org/document/9490512) | [Link](https://github.com/qdushl/Waterloo-Point-Cloud-Database-2.0)     

​                                           

### **Run Code for Different Models**

1. `python _{PCQA}_step1.py`
2. `python _{PCQA}_step2.py`

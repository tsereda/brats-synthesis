@article{jiang2025fast,
  title={Fast-DDPM: Fast denoising diffusion probabilistic models for medical image-to-image generation},
  author={Jiang, Hongxu and Imran, Muhammad and Zhang, Teng and Zhou, Yuyin and Liang, Muxuan and Gong, Kuang and Shao, Wei},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2025},
  publisher={IEEE}
}
@article{friedrich2024cwdm,
         title={cWDM: Conditional Wavelet Diffusion Models for Cross-Modality 3D Medical Image Synthesis},
         author={Paul Friedrich and Alicia Durrer and Julia Wolleb and Philippe C. Cattin},
         year={2024},
         journal={arXiv preprint arXiv:2411.17203}}
### Challenge Goal

The main goal of the **BraSyn Challenge** is to develop algorithms that can create, or "synthesize," a missing brain MRI scan when the other scans are available. Most advanced brain tumor segmentation algorithms require a complete set of four MRI modalities:

  * T1-weighted (T1)
  * T1-weighted with contrast-enhancement (T1c)
  * T2-weighted (T2w)
  * T2 Fluid-Attenuated Inversion Recovery (FLAIR)

This challenge addresses the common clinical problem where one of these scans might be missing, allowing powerful analysis tools to be used on incomplete or older datasets.

-----

### Your Task

When given a patient's data, one of the four MRI modalities will be randomly removed. Your algorithm's job is to take the **three available MRI volumes** and **generate the fourth, missing one**.

-----

### Data

The challenge uses a diverse dataset from multiple institutions, based on the **BraTS-GLI 2023**, **BraTS-METS 2023**, and **BraTS-Meningioma 2023** collections. This means you'll encounter variations in image quality, scanner types, and acquisition protocols.

  * **Final Datasets Released:** The final, quality-controlled training and validation datasets were released on **June 26, 2025**, and are ready for use. You can access them here: [BraTS-25 datasets](https://www.google.com/search?q=https://www.synapse.org/%23!Synapse:syn52631325/files/).
  * **Important Note:** During the validation and testing phases, you will **not** be given the ground truth tumor segmentation masks.

-----

### Assessment and Evaluation

Your synthesized images will be judged in two primary ways:

1.  **Image Quality:** The **Structural Similarity Index Measure (SSIM)** will be used to compare your generated image to the real one. This will be measured separately for the tumor area and the healthy brain tissue.
2.  **Segmentation Performance:** This is an indirect metric. Your synthesized image will be added back to the set of three originals. A pre-trained, state-of-the-art segmentation algorithm will then be run on this "complete" set of four images. The resulting tumor segmentation will be evaluated against the ground truth using **Dice scores** and the **Hausdorff distance**.

-----

### Submission Format

The submission format differs between the validation and final test stages. All files must be submitted in a single `.zip` or `.tar.gz` archive.

#### **Validation Stage Submission**

For validation, you submit the **segmentation masks** that you generate after running the provided segmentation tool on your completed image sets.

  * **Filename Format:** `{cohort}-{ID}-{timepoint}.nii.gz`
  * **Example:** `BraTS-GLI-00001-001.nii.gz`

#### **Final Test Stage Submission**

For the final test, you submit the **synthesized MR images** that your algorithm created.

  * **Filename Format:** `{cohort}-{ID}-{timepoint}-{missing_modality}.nii.gz`
  * **Example:** `BraTS-GLI-00001-001-t2f.nii.gz`
  * **Missing Modalities:** `t1c`, `t1n`, `t2f`, `t2w`
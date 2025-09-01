# DongbaMIE: A Multimodal Information Extraction Dataset for Evaluating Semantic Understanding of Dongba Pictograms

<p align="center"><img src="./figures/fig1_dongba.png" alt="Image" width=50% ></p>

<div align="center">
    <a href="https://arxiv.org/abs/2503.03644">📖<strong>arXiv</strong></a> | <a href="https://huggingface.co/datasets/thinklis/DongbaMIE">🤗<strong>Dataset</strong></a>
</div>


<p align="center"">
<img src="figures/dongbamie_emnlp2025.svg" alt="DongbaMIE Accepted!" width="600" height="75" />
</p>

  
## Timeline


<!-- <p align="left" style="font-size:16px;">
  🎉 
  <span style="background: linear-gradient(to right, #ff8c00, #28a745, #007bff, #800080, #daa520);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;">
    [2025-08-21] DongbaMIE was accepted by EMNLP 2025 Findings
  </span> 
  🎊🥳
</p> -->


📢 [2025-03-06] DongbaMIE dataset released.

📢 [2025-03-05] Paper and repo released.  

##  Constructing DongbaMIE Dataset

<p align="center"><img src="./figures/fig2_dataset.png" alt="Image" width=60% ></p>


## Project Overview
This repository contains the following files:

- **[generate_data_qwen2vl.py](./generate_data_qwen2vl.py)** for building qwen2-vl format data
- **[gpt4o_vqa_test.py](./gpt4o_vqa_test.py)** for getting gpt4o and gemini test results
- **[metric.py](./metric.py)** for performance evaluation


##  Semantic visualization result

<p align="center"><img src="./figures/fig3_graph.png" alt="Image" width=60% ></p>



## DongbaMIE dataset statistics

<p align="center"><img src="./figures/fig6_dataset_statistics.png" alt="Image" width=100% ></p>


## Result

####  Overall Performance
<p align="center"><img src="./figures/fig5_result1.png" alt="Image" width=100% ></p>

#### The results of extracting the four semantic dimensions of objects, actions, relations and attributes simultaneously in a single reasoning
<p align="center"><img src="./figures/fig5_result2.png" alt="Image" width=100% ></p>





## Citation
If you find our project useful, please consider citing:
```
@misc@misc{bi2025dongbamiemultimodalinformationextraction,
      title={DongbaMIE: A Multimodal Information Extraction Dataset for Evaluating Semantic Understanding of Dongba Pictograms}, 
      author={Xiaojun Bi and Shuo Li and Ziyue Wang and Fuwen Luo and Weizheng Qiao and Lu Han and Ziwei Sun and Peng Li and Yang Liu},
      year={2025},
      eprint={2503.03644},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.03644}, 
}
```

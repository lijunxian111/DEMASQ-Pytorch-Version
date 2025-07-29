This is the pytorch version reproduction for the ndss23 paper: DEMASQ; official code is not released by the authors, and I'm not sure if all my options in my reproduction is correct.    
dataset:  
`We use the dataset for kaggle competition 'LLM generated text detecting' and DAIGT V2 trainset. The two ones can  
be found on kaggle. You can also add your own dataset. Put it under the 'dataset' directory.`
`link: https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset https://www.kaggle.com/competitions/llm-detect-ai-generated-text`  
running:  
`In main.py, let 'mod' = 'DEMASQ' and you can run the DEMASQ`  
transformers for feature extracting:  
`We use one kind of distillbert. You can both download it locally and directly use the API from huggingface.`  
